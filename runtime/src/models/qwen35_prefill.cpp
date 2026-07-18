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
    // Projection scratch must cover every proj() shape, not just the FFN: on the MoE model ffn=512
    // is small but the GDN/attn projections (wqkv=lqkv, ssm_out K=lvdim, ...) are wider. Size the
    // int8/dequant scratch on the true maxima so proj() never overruns on either model.
    auto umax = [](size_t a, size_t b) { return a > b ? a : b; };
    const size_t maxw = umax(umax((size_t)ffn * H, (size_t)lqkv * H),
                             umax((size_t)H * lvdim, (size_t)H * qdim));   // largest weight tile
    const int max_k   = (int)umax(umax((size_t)H, (size_t)ffn), umax((size_t)lvdim, (size_t)qdim));
    const int max_out = (int)umax(umax((size_t)ffn, (size_t)lqkv), umax((size_t)H, (size_t)wide));
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
    bf16* att  = a.alloc<bf16>((size_t)N * lvdim);       // attn out / gdn_out (4096)
    bf16* lnrm = a.alloc<bf16>((size_t)N * lvdim);       // lin_norm (4096)
    bf16* la   = a.alloc<bf16>((size_t)N * vh);          // lin_alpha (32)
    bf16* lb   = a.alloc<bf16>((size_t)N * vh);          // lin_beta (32)
    bf16* ffg  = a.alloc<bf16>((size_t)N * ffn);         // ffn gate (12288)
    bf16* ffu  = a.alloc<bf16>((size_t)N * ffn);         // ffn up
    bf16* ffh  = a.alloc<bf16>((size_t)N * ffn);         // ffn silu(gate)*up
    bf16* wbuf = a.alloc<bf16>(maxw);                    // dequantized-weight scratch (reused)
    int*  d_ids = a.alloc<int>((size_t)N);
    if (!a.ok) { a.free_all(); fprintf(stderr, "[prefill] scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }
    // int8 tensor-core projections (prefill_gemm_i8): ~2x the bf16 GEMM at int8==bf16 output fidelity
    // (GGUF weights are already Q4_K/Q6_K -> int8 weight-quant is lossless vs what's stored). Default
    // ON at every batched context; SPARKINFER_PREFILL_I8=0 disables (A/B). The int8 scratch lives in
    // its own arena so an alloc failure at huge N degrades to the bf16 GEMMs, not to the token loop.
    const char* _pi8 = getenv("SPARKINFER_PREFILL_I8");
    bool use_i8 = !(_pi8 && _pi8[0] == '0');
    Arena a8;
    signed char* A_i8 = use_i8 ? a8.alloc<signed char>((size_t)N * max_k) : nullptr;
    signed char* W_i8 = use_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = use_i8 ? a8.alloc<float>((size_t)N) : nullptr;
    float* sw = use_i8 ? a8.alloc<float>((size_t)max_out) : nullptr;
    if (use_i8 && !a8.ok) { a8.free_all(); use_i8 = false; }

    // ---- grouped-MoE prefill scratch (Qwen3.6) ----------------------------------------------
    // One int8 requant of the whole [E,*] expert stack per weight per layer (reused across the
    // layer's tokens), plus the permuted-assignment activation/output buffers. Its own arena so an
    // OOM at large N degrades to the token loop instead of the dense-FFN path.
    const int E = c.n_experts, F = c.moe_ffn, TK = c.top_k;
    const int max_tiles = is_moe ? kernels::moe_prefill_max_tiles(N, TK, E) : 0;
    const int prows = max_tiles * 128;                   // padded permuted rows
    Arena am;
    float*  m_logits = nullptr; int* m_ids = nullptr; float* m_wts = nullptr; int* m_counts = nullptr;
    int*    m_off = nullptr; int* m_postok = nullptr; float* m_poswt = nullptr; int* m_tile = nullptr;
    signed char *m_Ai8 = nullptr, *m_Hi8 = nullptr, *m_Wg = nullptr, *m_Wu = nullptr, *m_Wd = nullptr;
    float   *m_sA = nullptr, *m_sH = nullptr, *m_sWg = nullptr, *m_sWu = nullptr, *m_sWd = nullptr;
    bf16    *m_G = nullptr, *m_U = nullptr, *m_D = nullptr, *m_rw = nullptr;
    float*  m_routed = nullptr;
    bf16    *m_shared = nullptr, *m_shg = nullptr, *m_shu = nullptr, *m_shh = nullptr;
    bf16    *m_sgw = nullptr, *m_suw = nullptr, *m_sdw = nullptr, *m_sgiw = nullptr;
    bf16    *m_wdq = nullptr;
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
        m_Ai8    = am.alloc<signed char>((size_t)prows * H);
        m_sA     = am.alloc<float>(prows);
        m_Wg     = am.alloc<signed char>((size_t)E * F * H); m_sWg = am.alloc<float>((size_t)E * F);
        m_Wu     = am.alloc<signed char>((size_t)E * F * H); m_sWu = am.alloc<float>((size_t)E * F);
        m_Wd     = am.alloc<signed char>((size_t)E * H * F); m_sWd = am.alloc<float>((size_t)E * H);
        m_G      = am.alloc<bf16>((size_t)prows * F);
        m_U      = am.alloc<bf16>((size_t)prows * F);
        m_Hi8    = am.alloc<signed char>((size_t)prows * F); m_sH = am.alloc<float>(prows);
        m_D      = am.alloc<bf16>((size_t)prows * H);
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
        m_wdq       = am.alloc<bf16>((size_t)E * F * H);   // bf16 fallback for unsupported quant (UD Q5_K)
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

        // h = x + ao ; hn = RMSNorm(h, post_attn_norm)
        kernels::launch_prefill_add(x, ao, hbuf, (long)N * H, st);
        kernels::launch_rmsnorm(hbuf, w.post_attn_norm, hn, N, H, eps, st);

        if (!is_moe) {
            // dense SwiGLU FFN
            proj(hn, w.gate_q, w.gate_qtype, ffg, ffn, H);
            proj(hn, w.up_q,   w.up_qtype,   ffu, ffn, H);
            kernels::launch_prefill_swiglu(ffg, ffu, ffh, (long)N * ffn, st);
            proj(ffh, w.down_q, w.down_qtype, ao, H, ffn);
            // x = h + ffn_out
            kernels::launch_prefill_add(hbuf, ao, x, (long)N * H, st);
        } else {
            // grouped int8 tensor-core MoE FFN: route -> permute into per-expert groups ->
            // gate/up/down as grouped int8 GEMMs -> weighted scatter + gated shared expert.
            kernels::launch_moe_prefill_router_logits(hn, w.router_w, w.router_w_type, m_rw,
                                                      m_logits, N, H, E, st);
            pf_cu(cudaMemsetAsync(m_counts, 0, (size_t)E * sizeof(int), st), "moe counts zero");
            kernels::launch_moe_router(m_logits, m_ids, m_wts, m_counts, N, E, TK, 1, st);
            kernels::launch_moe_prefill_build_groups(m_ids, m_wts, m_counts, m_off, m_postok,
                                                     m_poswt, m_tile, nullptr, N, TK, E, st);
            kernels::launch_moe_prefill_gather_quant_i8(hn, m_postok, TK, m_Ai8, m_sA, prows, H, st);
            // one int8 requant of the whole expert stack, reused across all of this layer's tokens.
            // Fused Q4_K/Q6_K -> int8; UD mixed quants (e.g. Q5_K down) fall back to dequant->bf16
            // (the validated GGUF path) then per-row int8, matching the dense proj() fallback.
            auto requant = [&](int qt, const void* Wq, signed char* Wi8, float* sWi8, int rows, int cols) {
                if (!kernels::launch_gguf_dequant_rows_i8(qt, Wq, Wi8, sWi8, rows, cols, st)) {
                    kernels::launch_gguf_dequant(qt, Wq, m_wdq, (long)rows * cols, st);
                    kernels::launch_prefill_quantize_rows_i8(m_wdq, Wi8, sWi8, rows, cols, st);
                }
            };
            requant(w.gate_qtype, w.gate_q, m_Wg, m_sWg, E * F, H);
            requant(w.up_qtype,   w.up_q,   m_Wu, m_sWu, E * F, H);
            requant(w.down_qtype, w.down_q, m_Wd, m_sWd, E * H, F);
            kernels::launch_moe_grouped_gemm_i8(m_Ai8, m_sA, m_Wg, m_sWg, m_off, m_tile, max_tiles, m_G, F, H, st);
            kernels::launch_moe_grouped_gemm_i8(m_Ai8, m_sA, m_Wu, m_sWu, m_off, m_tile, max_tiles, m_U, F, H, st);
            kernels::launch_moe_prefill_swiglu_quant_i8(m_G, m_U, m_Hi8, m_sH, prows, F, st);
            kernels::launch_moe_grouped_gemm_i8(m_Hi8, m_sH, m_Wd, m_sWd, m_off, m_tile, max_tiles, m_D, H, F, st);
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
            // x = h + routed + sigmoid(glogit) * shared
            kernels::launch_moe_prefill_finalize(hbuf, m_routed, shared_out, glogit, x, N, H, st);
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
