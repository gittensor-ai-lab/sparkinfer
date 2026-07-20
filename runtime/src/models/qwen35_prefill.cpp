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
#include "sparkinfer/kernels/moe_prefill_grouped.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace sparkinfer {

namespace {
using bf16 = unsigned short;
inline void pf_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[prefill] %s: %s\n", what, cudaGetErrorString(e));
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
    // Qwen3.5 dense-hybrid, or the Qwen3.6-35B-A3B MoE hybrid (grouped int8 expert FFN below).
    const bool is_moe = !c.dense_ffn;
    if (!s.gguf || !c.hybrid || n <= 0) return -1;
    if (c.head_dim != 256 || c.linear_head_dim != 128) return -1;   // kernels specialize these
    if (is_moe && !(c.n_experts == 256 && c.top_k == 8 && c.moe_ffn == 512 && c.hidden == 2048))
        return -1;   // grouped MoE prefill specialized to the Qwen3.6-35B-A3B shape

    const int H = c.hidden;
    const int N = n;
    cudaStream_t st = s.stream;

    const int qdim = s.qdim, kvdim = s.kvdim;            // full-attn: 4096 / 1024
    const int lqkv = s.linear_qkvdim;                    // 8192
    const int lvdim = s.linear_vdim;                     // 4096
    const int vh   = c.linear_v_heads;                   // 32
    const int ffn  = c.moe_ffn;                          // dense: 12288 · MoE: 512 (per expert)
    const int wide = 2 * qdim;                           // 8192 (qraw); also >= lqkv
    // The dense FFN is processed in token-chunks so its ffn-wide scratch (ffg/ffu/A_i8) stays O(chunk)
    // instead of O(N) — at long context those full-width buffers dominate and OOM (~8 GB @128k). The
    // FFN is per-token independent, so chunking is numerically identical. Env override; default 32768.
    const int ffn_chunk = []{ const char* e = getenv("SPARKINFER_PREFILL_FFN_CHUNK"); int c = e ? atoi(e) : 32768; return c > 0 ? c : 32768; }();
    const int FC = (N < ffn_chunk) ? N : ffn_chunk;
    // Projection scratch must cover every proj() shape, not just the FFN: on the MoE model ffn=512
    // is small but the GDN/attn projections (wqkv=lqkv, ssm_out K=lvdim, ...) are wider. Size the
    // int8/dequant scratch on the true maxima so proj() never overruns on either model. The dense FFN
    // (K=ffn) is chunked below, so max_k covers only the full-N projections (attn/GDN, K<=lvdim).
    auto umax = [](size_t a, size_t b) { return a > b ? a : b; };
    const size_t maxw = umax(umax((size_t)ffn * H, (size_t)lqkv * H),
                             umax((size_t)H * lvdim, (size_t)H * qdim));   // largest weight tile
    const int max_k   = (int)umax((size_t)H, umax((size_t)lvdim, (size_t)qdim));
    const int max_out = (int)umax(umax((size_t)ffn, (size_t)lqkv), umax((size_t)H, (size_t)wide));
    bf16* lin_conv_state = static_cast<bf16*>(s.lin_conv_state);

    // ---- scratch ----
    Arena a;
    bf16* x    = a.alloc<bf16>((size_t)N * H);
    bf16* xn   = a.alloc<bf16>((size_t)N * H);
    bf16* hn   = a.alloc<bf16>((size_t)N * H);
    bf16* ao   = a.alloc<bf16>((size_t)N * H);
    bf16* b8   = a.alloc<bf16>((size_t)N * wide);        // qraw / lin_qkv (8192)
    bf16* lz   = a.alloc<bf16>((size_t)N * lvdim);       // lin_z (4096)
    bf16* gq   = a.alloc<bf16>((size_t)N * s.linear_qdim);   // gdn q (2048)
    bf16* gk   = a.alloc<bf16>((size_t)N * s.linear_qdim);   // gdn k (2048)
    bf16* gv   = a.alloc<bf16>((size_t)N * lvdim);       // gdn v (4096)
    bf16* att  = a.alloc<bf16>((size_t)N * lvdim);       // attn out / gdn_out (4096)
    bf16* lnrm = a.alloc<bf16>((size_t)N * lvdim);       // lin_norm (4096)
    bf16* la   = a.alloc<bf16>((size_t)N * vh);          // lin_alpha (32)
    bf16* lb   = a.alloc<bf16>((size_t)N * vh);          // lin_beta (32)
    // Full-attention scratch ALIASES the GDN scratch: a layer is either linear-attn (GDN) or full
    // softmax-attn, never both, and qb/qg/kf/vf are pairwise-distinct within a full-attn layer while
    // the GDN buffers they map onto are unused there (and vice-versa). Saves ~10K bf16/token of peak
    // scratch at long context (each is <= its GDN host: qdim/kvdim <= lvdim/linear_qdim).
    bf16* qb   = gv;                                     // full q      (4096) <- gdn v    (4096)
    bf16* qg   = lnrm;                                   // full q-gate (4096) <- lin_norm (4096)
    bf16* kf   = gq;                                     // full k      (1024) <- gdn q    (2048)
    bf16* vf   = gk;                                     // full v      (1024) <- gdn k    (2048)
    bf16* ffg  = a.alloc<bf16>((size_t)FC * ffn);        // ffn gate (12288), bounded to FC tokens
    bf16* ffu  = a.alloc<bf16>((size_t)FC * ffn);        // ffn up,          bounded to FC tokens
    bf16* ffh  = ffg;                                    // SwiGLU computed in-place into ffg (down reads it)
    bf16* wbuf = a.alloc<bf16>(maxw);                    // dequantized-weight scratch (reused)
    int*  d_ids = a.alloc<int>((size_t)N);
    if (!a.ok) { a.free_all(); fprintf(stderr, "[prefill] scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }
    // int8 tensor-core projections (prefill_gemm_i8): ~2x the bf16 GEMM at int8==bf16 output fidelity
    // (GGUF weights are already Q4_K/Q6_K -> int8 weight-quant is lossless vs what's stored). Default
    // ON at every batched context; SPARKINFER_PREFILL_I8=0 disables (A/B). The int8 scratch lives in
    // its own arena so an alloc failure at huge N degrades to the bf16 GEMMs, not to the token loop.
    const char* _pi8 = getenv("SPARKINFER_PREFILL_I8");
    bool use_i8 = !(_pi8 && _pi8[0] == '0');
    // Long-context fidelity: the near-1-decay GDN recurrence amplifies the per-row int8
    // activation-quant error across the sequence, so int8 prefill diverges from the token-by-token
    // path past ~96k (128k: top1 0.31 / KL 0.18). Above bf16_minctx (default 96k) fall back to bf16
    // projections, which stay faithful (128k: top1 0.69 / KL 0.04) at ~half the prefill throughput —
    // still ~26x the sequential token loop. Short/mid contexts keep int8 (full speed, unchanged).
    // SPARKINFER_PREFILL_BF16_MINCTX overrides the threshold.
    static int bf16_minctx = []{ const char* e = getenv("SPARKINFER_PREFILL_BF16_MINCTX"); return e ? atoi(e) : 98304; }();
    if (N > bf16_minctx) use_i8 = false;
    Arena a8;
    // A_i8 holds the quantized activation: full-N projs quantize N rows x K(<=max_k); the chunked
    // dense FFN quantizes at most FC rows x ffn. Size to the max of the two.
    const size_t a_i8_sz = umax((size_t)N * max_k, (size_t)FC * ffn);
    signed char* A_i8 = use_i8 ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    signed char* W_i8 = use_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = use_i8 ? a8.alloc<float>((size_t)N) : nullptr;
    float* sw = use_i8 ? a8.alloc<float>((size_t)max_out) : nullptr;
    if (use_i8 && !a8.ok) { a8.free_all(); use_i8 = false; }

    // ---- grouped-MoE prefill scratch (Qwen3.6) ----------------------------------------------
    // Permuted-assignment buffers + the gate/up and down outputs. Expert weights are read native
    // (faithful K-quant dequant in the kernel), so no int8 requant scratch. Own arena so an OOM at
    // large N degrades to the token loop instead of the dense-FFN path.
    const int E = c.n_experts, F = c.moe_ffn, TK = c.top_k;
    const int max_tiles = is_moe ? kernels::moe_prefill_max_tiles(N, TK, E) : 0;
    const int prows = is_moe ? kernels::moe_prefill_padded_rows(N, TK, E) : 0;   // padded permuted rows
    Arena am;
    float*  m_logits = nullptr; int* m_ids = nullptr; float* m_wts = nullptr; int* m_counts = nullptr;
    int*    m_off = nullptr; int* m_postok = nullptr; float* m_poswt = nullptr; int* m_tile = nullptr;
    bf16    *m_H = nullptr, *m_D = nullptr, *m_rw = nullptr;
    float*  m_routed = nullptr;
    bf16    *m_shared = nullptr, *m_shg = nullptr, *m_shu = nullptr, *m_shh = nullptr;
    bf16    *m_sgw = nullptr, *m_suw = nullptr, *m_sdw = nullptr, *m_sgiw = nullptr;
    float*  m_gate_logit = nullptr;
    if (is_moe) {
        m_logits = am.alloc<float>((size_t)N * E);
        m_ids    = am.alloc<int>((size_t)N * TK);
        m_wts    = am.alloc<float>((size_t)N * TK);
        m_counts = am.alloc<int>(E);
        m_off    = am.alloc<int>(2 * E + 1);             // offsets[E+1] + E scatter cursors
        m_postok = am.alloc<int>(prows);
        m_poswt  = am.alloc<float>(prows);
        m_tile   = am.alloc<int>(max_tiles);
        m_H      = am.alloc<bf16>((size_t)prows * F);    // silu(gate)*up
        m_D      = am.alloc<bf16>((size_t)prows * H);    // down output
        m_rw     = am.alloc<bf16>((size_t)E * H);
        m_routed = am.alloc<float>((size_t)N * H);
        m_shared = am.alloc<bf16>((size_t)N * H);
        m_shg    = am.alloc<bf16>((size_t)N * F);
        m_shu    = am.alloc<bf16>((size_t)N * F);
        m_shh    = am.alloc<bf16>((size_t)N * F);
        m_sgw    = am.alloc<bf16>((size_t)F * H);
        m_suw    = am.alloc<bf16>((size_t)F * H);
        m_sdw    = am.alloc<bf16>((size_t)H * F);
        m_sgiw   = am.alloc<bf16>(H);
        m_gate_logit = am.alloc<float>(N);
        if (!am.ok) { am.free_all(); a.free_all(); a8.free_all();
            fprintf(stderr, "[prefill] MoE scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }
    }

    pf_cu(cudaMemcpyAsync(d_ids, prompt_ids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "prefill ids");

    // Dequantize a native GGUF weight [n_out,K] to bf16 scratch; return a bf16 [n_out,K] ptr.
    auto dq = [&](const void* W, int wtype, int n_out, int K) -> const void* {
        if (wtype == 0) return W;   // already bf16 dense
        kernels::launch_gguf_dequant(wtype, W, wbuf, (long)n_out * K, st);
        return wbuf;
    };
    // C[N,n_out] = A[N,K] @ W^T  (W native quantized [n_out,K]).
    auto proj = [&](const bf16* A, const void* W, int wtype, bf16* C, int n_out, int K, int rows = 0) {
        const int R = rows > 0 ? rows : N;   // rows (M) to process; chunked FFN passes a sub-N count
        // int8 only for the big weight-bound projections; keep the tiny per-v-head gate
        // projections (ssm_alpha/ssm_beta, n_out == v_heads) in bf16 — they feed the GDN
        // sigmoid gates, where per-row int8 quant of a 32-wide weight costs more accuracy
        // than the negligible time it saves.
        if (use_i8 && n_out >= 128) {
            kernels::launch_prefill_quantize_rows_i8(A, A_i8, sx, R, K, st);
            // fused Q4_K/Q6_K -> int8 rows skips the dequant-to-bf16 scratch round trip
            if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
                const void* wb = dq(W, wtype, n_out, K);
                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
            }
            kernels::launch_prefill_gemm_i8(A_i8, W_i8, sx, sw, C, R, n_out, K, st);
        } else {
            kernels::launch_prefill_gemm(A, dq(W, wtype, n_out, K), C, R, n_out, K, st);
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

    // Return a bf16 [rows,cols] view of a shared-expert weight: pass through the bf16 dense tensor
    // if present, else dequant the GGUF-quantized tensor (Q8_0/Q4_K) into the provided scratch.
    auto sh_bf16 = [&](const void* wq, int qtype, const void* wbf, bf16* scratch, long nv) -> const bf16* {
        if (wbf) return reinterpret_cast<const bf16*>(wbf);
        if (wq)  { kernels::launch_gguf_dequant(qtype, wq, scratch, nv, st); return scratch; }
        return nullptr;
    };

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
            kernels::launch_prefill_attn_int8_paged(qb, kpool, vpool, kscale, vscale, btable, att,
                N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale, st);
            kernels::launch_prefill_mul_sigmoid(att, qg, N, qdim, st);
            proj(att, w.wo, w.wo_type, ao, H, qdim);
        }

        // x += ao (post-attn residual, in-place: hbuf folded into x) ; hn = RMSNorm(x, post_attn_norm)
        kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
        kernels::launch_rmsnorm(x, w.post_attn_norm, hn, N, H, eps, st);

        if (!is_moe) {
            // dense SwiGLU FFN, chunked over tokens: ffg/ffu/A_i8 stay O(FC*ffn). Per-token
            // independent, so numerically identical to the full-width pass; only the scratch shrinks.
            for (int fo = 0; fo < N; fo += FC) {
                const int fn = (N - fo < FC) ? (N - fo) : FC;
                const bf16* hn_c = hn + (size_t)fo * H;
                proj(hn_c, w.gate_q, w.gate_qtype, ffg, ffn, H, fn);
                proj(hn_c, w.up_q,   w.up_qtype,   ffu, ffn, H, fn);
                kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                proj(ffg, w.down_q, w.down_qtype, ao + (size_t)fo * H, H, ffn, fn);
            }
            kernels::launch_prefill_add(x, ao, x, (long)N * H, st);   // x += ffn_out (x already = h)
        } else {
            // faithful grouped MoE FFN: route -> permute into per-expert groups -> weight-stationary
            // gate/up + down (native K-quant dequant, byte-faithful) -> weighted scatter + shared.
            kernels::launch_moe_prefill_router_logits(hn, w.router_w, w.router_w_type, m_rw,
                                                      m_logits, N, H, E, st);
            pf_cu(cudaMemsetAsync(m_counts, 0, (size_t)E * sizeof(int), st), "moe counts zero");
            kernels::launch_moe_router(m_logits, m_ids, m_wts, m_counts, N, E, TK, 1, st);
            kernels::launch_moe_prefill_build_groups(m_ids, m_wts, m_counts, m_off, m_postok,
                                                     m_poswt, m_tile, nullptr, N, TK, E, st);
            kernels::launch_moe_grouped_gate_up(hn, m_postok, TK, m_tile, max_tiles,
                w.gate_q, w.up_q, w.gate_qtype, w.up_qtype, m_H, H, F, prows, st);
            kernels::launch_moe_grouped_down(m_H, m_postok, m_tile, max_tiles,
                w.down_q, w.down_qtype, m_D, H, F, prows, st);
            kernels::launch_moe_prefill_scatter_weighted(m_D, m_postok, TK, m_poswt, m_routed, prows, N, H, st);
            // shared expert (Q8_0/Q4_K -> bf16, batched bf16 GEMM) + its sigmoid scalar gate
            const bf16* sgw = sh_bf16(w.shared_gate_q, w.shared_gate_qtype, w.shared_gate, m_sgw, (long)F * H);
            const bf16* suw = sh_bf16(w.shared_up_q,   w.shared_up_qtype,   w.shared_up,   m_suw, (long)F * H);
            const bf16* sdw = sh_bf16(w.shared_down_q, w.shared_down_qtype, w.shared_down, m_sdw, (long)H * F);
            const bf16* shared_out = nullptr; const float* glogit = nullptr;
            if (c.n_shared > 0 && sgw && suw && sdw) {
                kernels::launch_prefill_gemm(hn, sgw, m_shg, N, F, H, st);
                kernels::launch_prefill_gemm(hn, suw, m_shu, N, F, H, st);
                kernels::launch_prefill_swiglu(m_shg, m_shu, m_shh, (long)N * F, st);
                kernels::launch_prefill_gemm(m_shh, sdw, m_shared, N, H, F, st);
                shared_out = m_shared;
                if (w.shared_gate_inp) {   // reuse the router-logits kernel as a single-expert dot
                    kernels::launch_moe_prefill_router_logits(hn, w.shared_gate_inp,
                        w.shared_gate_inp_type, m_sgiw, m_gate_logit, N, H, 1, st);
                    glogit = m_gate_logit;
                }
            }
            // x = h + routed + sigmoid(glogit) * shared  (in-place: x already = h post-attention)
            kernels::launch_moe_prefill_finalize(x, m_routed, shared_out, glogit, x, N, H, st);
        }

        // xn = RMSNorm(x, next_input_norm)  (final_norm on the last layer)
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
    am.free_all();
    return seed;
}

} // namespace sparkinfer
