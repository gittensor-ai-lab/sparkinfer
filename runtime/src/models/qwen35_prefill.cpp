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
#include "sparkinfer/kernels/prefill_moe.h"
#include "sparkinfer/kernels/moe.h"

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
    // Batched prefill supports the Qwen3.5 dense-hybrid (Qwythos) AND the Qwen3.6-35B-A3B MoE hybrid.
    // Both share the GDN + full-attention batched kernels (identical math at 128/16/32 GDN dims and
    // 256/64 attn dims); they differ ONLY in the FFN, branched below (dense SwiGLU vs the expert-
    // grouped int8 MoE path). The MoE path is specialized for 256 experts with a top-k router.
    const bool moe = !c.dense_ffn && c.n_experts > 0;
    if (!s.gguf || !c.hybrid || n <= 0) return -1;
    if (!c.dense_ffn && !moe) return -1;
    if (c.head_dim != 256 || c.linear_head_dim != 128) return -1;   // kernels specialize these
    if (moe && (c.n_experts != 256 || c.top_k <= 0)) return -1;     // grouped top-k path specialized for 256
    if (moe)
        for (int L = 0; L < c.n_layers; L++) {
            const Qwen35LayerWeights& w = s.w.layers[L];
            // grouped expert GEMMs need quantized experts (Q4_K/Q5_K/Q6_K rows-int8 dequant) + a router
            if (!w.gate_q || !w.up_q || !w.down_q || !w.router_w) {
                fprintf(stderr, "[prefill-moe] layer %d missing expert/router tensors -> token loop\n", L);
                return -1;
            }
            auto qok = [](int t) { return t == 12 || t == 13 || t == 14; };
            if (!qok(w.gate_qtype) || !qok(w.up_qtype) || !qok(w.down_qtype)) {
                fprintf(stderr, "[prefill-moe] layer %d expert qtypes %d/%d/%d unsupported -> token loop\n",
                        L, w.gate_qtype, w.up_qtype, w.down_qtype);
                return -1;
            }
        }

    const int H = c.hidden;
    const int N = n;
    cudaStream_t st = s.stream;

    const int qdim = s.qdim, kvdim = s.kvdim;            // full-attn: 4096 / 1024
    const int lqkv = s.linear_qkvdim;                    // 8192
    const int lvdim = s.linear_vdim;                     // 4096
    const int vh   = c.linear_v_heads;                   // 32
    const int ffn  = c.moe_ffn;                          // dense: 12288; MoE: per-expert 512
    const int wide = 2 * qdim;                           // 8192 (qraw); also >= lqkv
    // wbuf must hold the largest weight the `dq` lambda dequantizes: the dense FFN (ffn*H) OR, on the
    // MoE path (small ffn=512), the biggest projection (wide/lqkv * H). Cover all of them.
    size_t maxw = (size_t)wide * H;
    if ((size_t)lqkv * H > maxw) maxw = (size_t)lqkv * H;
    if (!moe && (size_t)ffn * H > maxw) maxw = (size_t)ffn * H;
    // int8 proj scratch dims: largest projection input K (A rows) and output n_out (channel scales).
    // On MoE the small per-expert ffn (512) is NOT the max, so size against the real projections.
    auto imax = [](int x, int y) { return x > y ? x : y; };
    const int maxAK = moe ? imax(qdim, lvdim) : imax(ffn, imax(qdim, lvdim));   // max proj input dim
    const int maxNO = moe ? imax(wide, lqkv) : imax(ffn, imax(wide, lqkv));     // max proj output dim
    // Dense FFN is processed in token-chunks so its ffn-wide scratch (ffg/ffu/A_i8) stays O(chunk)
    // instead of O(N) — at long context those full-width buffers dominate and OOM (~8 GB @128k). The
    // FFN is per-token independent, so chunking is numerically identical. Env override; default 32768.
    // (MoE doesn't use ffg/ffu — its grouped FFN has its own O(N*top_k) scratch, so chunking is moot.)
    const int ffn_chunk = []{ const char* e = getenv("SPARKINFER_PREFILL_FFN_CHUNK"); int c = e ? atoi(e) : 32768; return c > 0 ? c : 32768; }();
    const int FC = (N < ffn_chunk) ? N : ffn_chunk;
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
    // Dense: int8 projections default ON. MoE: default OFF — the discrete top-k router amplifies the
    // per-token int8 projection error into different expert selections, which diverges from the
    // token-by-token path far more than in the dense FFN; bf16 projections keep the batched MoE
    // prefill faithful to the decode path. SPARKINFER_PREFILL_I8 overrides either way.
    bool use_i8 = _pi8 ? (_pi8[0] != '0') : !moe;
    // Long-context fidelity (dense): the near-1-decay GDN recurrence amplifies the per-row int8
    // activation-quant error across the sequence, so int8 prefill diverges from the token-by-token
    // path past ~96k (128k: top1 0.31 / KL 0.18). Above bf16_minctx (default 96k) fall back to bf16
    // for GDN/attn projections. The dense FFN is per-token (no recurrence), so it can stay on the
    // int8 tensor-core path — recovering most of the ~2x cliff (18k→8.5k pp) without GDN drift.
    // SPARKINFER_PREFILL_BF16_MINCTX overrides the threshold; SPARKINFER_PREFILL_I8_FFN=0 disables
    // the selective FFN-int8 recovery (A/B).
    static int bf16_minctx = []{ const char* e = getenv("SPARKINFER_PREFILL_BF16_MINCTX"); return e ? atoi(e) : 98304; }();
    const bool long_bf16 = !moe && N > bf16_minctx;
    if (long_bf16) use_i8 = false;
    const char* _pi8ffn = getenv("SPARKINFER_PREFILL_I8_FFN");
    bool use_i8_ffn = long_bf16 && (!_pi8ffn || _pi8ffn[0] != '0');
    // Full-attn Q/K/V/O are also per-token (no GDN recurrence). Keep them on int8 at long ctx
    // unless SPARKINFER_PREFILL_I8_ATTN=0. GDN projections always stay bf16 above bf16_minctx.
    const char* _pi8attn = getenv("SPARKINFER_PREFILL_I8_ATTN");
    bool use_i8_attn = long_bf16 && (!_pi8attn || _pi8attn[0] != '0');
    Arena a8;
    // A_i8 holds the quantized activation. Dense full-i8: non-FFN projs quantize N rows x K(<=H);
    // chunked FFN quantizes at most FC rows x ffn. Long-ctx selective: N*H if attn-i8 else FC*ffn.
    // MoE: no chunked FFN; projections quantize N rows x maxAK.
    const bool need_i8 = use_i8 || use_i8_ffn || use_i8_attn;
    const size_t a_i8_sz = moe ? (size_t)N * maxAK
                               : ((use_i8 || use_i8_attn)
                                  ? (((size_t)N * H > (size_t)FC * ffn) ? (size_t)N * H : (size_t)FC * ffn)
                                  : (size_t)FC * ffn);
    const size_t sx_n = (use_i8 || use_i8_attn) ? (size_t)N : (size_t)FC;
    signed char* A_i8 = need_i8 ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    signed char* W_i8 = need_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = need_i8 ? a8.alloc<float>(sx_n) : nullptr;
    float* sw = need_i8 ? a8.alloc<float>((size_t)maxNO) : nullptr;
    if (need_i8 && !a8.ok) {
        a8.free_all();
        use_i8 = false;
        use_i8_ffn = false;
        use_i8_attn = false;
        A_i8 = W_i8 = nullptr;
        sx = sw = nullptr;
    }

    // Long-ctx FFN int8: keep gate/up/down int8 weights (+scales) across token chunks so each
    // layer dequants once instead of once per chunk. ~150 MB vs ~300 MB for a bf16 cache.
    Arena aw;
    signed char *ffn_Wg_i8 = nullptr, *ffn_Wu_i8 = nullptr, *ffn_Wd_i8 = nullptr;
    float *ffn_swg = nullptr, *ffn_swu = nullptr, *ffn_swd = nullptr;
    if (use_i8_ffn) {
        ffn_Wg_i8 = aw.alloc<signed char>((size_t)ffn * H);
        ffn_Wu_i8 = aw.alloc<signed char>((size_t)ffn * H);
        ffn_Wd_i8 = aw.alloc<signed char>((size_t)H * ffn);
        ffn_swg = aw.alloc<float>((size_t)ffn);
        ffn_swu = aw.alloc<float>((size_t)ffn);
        ffn_swd = aw.alloc<float>((size_t)H);
        if (!aw.ok) {
            aw.free_all();
            ffn_Wg_i8 = ffn_Wu_i8 = ffn_Wd_i8 = nullptr;
            ffn_swg = ffn_swu = ffn_swd = nullptr;
            use_i8_ffn = false;
        }
    }

    // ---- MoE (Qwen3.6) scratch: expert-int8 weights + pair bucketing + pair-major hidden ----
    // The expert-grouped GEMMs run int8 tensor-core UNCONDITIONALLY (that is the speedup), so this
    // block carries its own int8 activation scratch (mA_i8/msx) and does not depend on the shared
    // `use_i8` flag, which upstream defaults OFF for MoE (it governs only the bf16-vs-int8 choice of
    // the attention/GDN/shared projections routed through `proj`). Full-N shared-expert buffers
    // (sfg/sfu/sfh) are dedicated here because the outer ffg/ffu are FC-chunked (dense path only).
    const int E = moe ? c.n_experts : 0, topk = moe ? c.top_k : 0, mffn = moe ? c.moe_ffn : 0;
    const int P = moe ? N * topk : 0;                          // routed (token, expert) pairs
    const int max_tiles = moe ? (P + 127) / 128 + E : 0;       // per-expert 128-row GEMM tiles
    Arena am;
    signed char *Wg_i8 = nullptr, *Wu_i8 = nullptr, *Wd_i8 = nullptr, *h_i8 = nullptr, *mA_i8 = nullptr;
    float *swg = nullptr, *swu = nullptr, *swd = nullptr, *sh = nullptr, *msx = nullptr;
    float *mlogits = nullptr, *mweights = nullptr, *pair_w = nullptr, *routed_f32 = nullptr, *dw = nullptr;
    int *mids = nullptr, *mcounts = nullptr, *moffsets = nullptr, *mcursors = nullptr;
    int *pair_tok = nullptr, *tilemap = nullptr, *d_ntiles = nullptr;
    bf16 *hg = nullptr, *hu = nullptr, *hh = nullptr, *sfg = nullptr, *sfu = nullptr, *sfh = nullptr;
    if (moe) {
        Wg_i8 = am.alloc<signed char>((size_t)E * mffn * H);
        Wu_i8 = am.alloc<signed char>((size_t)E * mffn * H);
        Wd_i8 = am.alloc<signed char>((size_t)E * H * mffn);
        swg = am.alloc<float>((size_t)E * mffn);
        swu = am.alloc<float>((size_t)E * mffn);
        swd = am.alloc<float>((size_t)E * H);
        mlogits = am.alloc<float>((size_t)N * E);
        mids = am.alloc<int>((size_t)P);
        mweights = am.alloc<float>((size_t)P);
        mcounts = am.alloc<int>(E);
        moffsets = am.alloc<int>(E + 1);
        mcursors = am.alloc<int>(E);
        pair_tok = am.alloc<int>((size_t)P);
        pair_w = am.alloc<float>((size_t)P);
        tilemap = am.alloc<int>((size_t)2 * max_tiles);
        d_ntiles = am.alloc<int>(1);
        hg = am.alloc<bf16>((size_t)P * mffn);
        hu = am.alloc<bf16>((size_t)P * mffn);
        hh = am.alloc<bf16>((size_t)P * mffn);
        h_i8 = am.alloc<signed char>((size_t)P * mffn);
        sh = am.alloc<float>((size_t)P);
        routed_f32 = am.alloc<float>((size_t)N * H);
        dw = am.alloc<float>((size_t)N);
        mA_i8 = am.alloc<signed char>((size_t)N * H);          // int8 activation for the grouped GEMMs
        msx = am.alloc<float>((size_t)N);
        sfg = am.alloc<bf16>((size_t)N * mffn);                // shared-expert gate/up/hidden (full N)
        sfu = am.alloc<bf16>((size_t)N * mffn);
        sfh = am.alloc<bf16>((size_t)N * mffn);
        if (!am.ok) {
            a.free_all(); a8.free_all(); am.free_all(); aw.free_all();
            fprintf(stderr, "[prefill] MoE scratch alloc failed (ctx=%d) -> fallback\n", N);
            return -1;
        }
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
            // mma.sync bf16 GEMM only for dense-hybrid long prefill (the >96k int8→bf16 fallback).
            // MoE always stays on wmma: its top-k router turns tiny GEMM differences into expert
            // flips that fail the Qwen3.6 accuracy gate.
            // Gate on full prompt length N (not chunk rows R): FFN is token-chunked to FC=32k for
            // VRAM, so R<=FC would otherwise keep the dominant gate/up/down GEMMs on wmma forever.
            const bool prefer_mma = !moe && N > bf16_minctx;
            kernels::launch_prefill_gemm(A, dq(W, wtype, n_out, K), C, R, n_out, K, st, prefer_mma);
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
            // Long-ctx: optionally keep Q/K/V/O on int8 (no GDN recurrence here).
            const bool restore_i8 = use_i8;
            if (use_i8_attn) use_i8 = true;
            proj(xn, w.wq, w.wq_type, b8, wide,  H);                 // qraw = [q|gate] per head
            proj(xn, w.wk, w.wk_type, kf, kvdim, H);
            proj(xn, w.wv, w.wv_type, vf, kvdim, H);
            kernels::launch_prefill_split_q_gate(b8, qb, qg, N, c.n_q_heads, c.head_dim, st);
            signed char* kpool = (signed char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            signed char* vpool = (signed char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            if (!kv8) { a.free_all(); a8.free_all(); am.free_all(); aw.free_all(); fprintf(stderr, "[prefill] batched prefill requires int8 KV\n"); return -1; }
            kernels::launch_prefill_qknorm_rope_kv_int8(qb, kf, vf, w.q_norm, w.k_norm,
                kpool, vpool, kscale, vscale, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                rope_dim, rope_theta, eps, bs, mbs, st);
            kernels::launch_prefill_attn_int8_paged(qb, kpool, vpool, kscale, vscale, btable, att,
                N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale, st);
            kernels::launch_prefill_mul_sigmoid(att, qg, N, qdim, st);
            proj(att, w.wo, w.wo_type, ao, H, qdim);
            use_i8 = restore_i8;
        }

        // x += ao (post-attn residual, in-place) ; hn = RMSNorm(x, post_attn_norm)
        kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
        kernels::launch_rmsnorm(x, w.post_attn_norm, hn, N, H, eps, st);

        if (!moe) {
            // dense SwiGLU FFN, chunked over tokens (upstream #530): ffg/ffu/A_i8 stay O(FC*ffn).
            // Per-token independent, so this is numerically identical to the full-width pass.
            // Long-ctx: selective int8 FFN (GDN/attn stay bf16) + int8 weight cache across chunks.
            const bool ffn_i8 = use_i8_ffn && ffn_Wg_i8 != nullptr;
            auto dequant_w_i8 = [&](int wtype, const void* W, signed char* dst, float* scale,
                                    int n_out, int K) {
                if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, dst, scale, n_out, K, st)) {
                    const void* wb = dq(W, wtype, n_out, K);
                    kernels::launch_prefill_quantize_rows_i8(wb, dst, scale, n_out, K, st);
                }
            };
            if (ffn_i8) {
                dequant_w_i8(w.gate_qtype, w.gate_q, ffn_Wg_i8, ffn_swg, ffn, H);
                dequant_w_i8(w.up_qtype,   w.up_q,   ffn_Wu_i8, ffn_swu, ffn, H);
                dequant_w_i8(w.down_qtype, w.down_q, ffn_Wd_i8, ffn_swd, H, ffn);
            }
            for (int fo = 0; fo < N; fo += FC) {
                const int fn = (N - fo < FC) ? (N - fo) : FC;
                const bf16* hn_c = hn + (size_t)fo * H;
                if (ffn_i8) {
                    kernels::launch_prefill_quantize_rows_i8(hn_c, A_i8, sx, fn, H, st);
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wg_i8, sx, ffn_swg, ffg, fn, ffn, H, st);
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wu_i8, sx, ffn_swu, ffu, fn, ffn, H, st);
                    kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                    kernels::launch_prefill_quantize_rows_i8(ffg, A_i8, sx, fn, ffn, st);
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wd_i8, sx, ffn_swd,
                                                    ao + (size_t)fo * H, fn, H, ffn, st);
                } else {
                    proj(hn_c, w.gate_q, w.gate_qtype, ffg, ffn, H, fn);
                    proj(hn_c, w.up_q,   w.up_qtype,   ffu, ffn, H, fn);
                    kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                    proj(ffg, w.down_q, w.down_qtype, ao + (size_t)fo * H, H, ffn, fn);
                }
            }
            // x += ffn_out (post-attn residual already folded into x above)
            kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
        } else {
            // ---- expert-grouped 256-expert int8 MoE FFN (this PR): route -> bucket routed
            // (token, expert) pairs by expert -> per-expert int8 tensor-core GEMMs, so each expert's
            // weights are read ONCE per layer instead of once per routed token (the ~1.1 GB/token
            // MoE weight re-read that pinned the token loop). Router logits use the decode-reference
            // gemv_f32-order dot; the router weight may itself be quantized in the UD GGUF. ----
            const void* rw = w.router_w_type ? dq(w.router_w, w.router_w_type, E, H) : w.router_w;
            kernels::launch_pfm_router_logits(hn, rw, mlogits, N, E, H, st);
            pf_cu(cudaMemsetAsync(mcounts, 0, E * sizeof(int), st), "moe counts zero");
            kernels::launch_moe_router(mlogits, mids, mweights, mcounts, N, E, topk, 1, st);
            kernels::launch_pfm_bucket_pairs(mids, mweights, mcounts, moffsets, mcursors,
                                             pair_tok, pair_w, tilemap, d_ntiles, N, E, topk, st);
            // Expert weights -> int8 rows ONCE per layer (one launch covers all 256 experts).
            kernels::launch_gguf_dequant_rows_i8(w.gate_qtype, w.gate_q, Wg_i8, swg, E * mffn, H, st);
            kernels::launch_gguf_dequant_rows_i8(w.up_qtype,   w.up_q,   Wu_i8, swu, E * mffn, H, st);
            kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q, Wd_i8, swd, E * H, mffn, st);
            kernels::launch_prefill_quantize_rows_i8(hn, mA_i8, msx, N, H, st);
            kernels::launch_pfm_moe_gemm_i8(mA_i8, msx, Wg_i8, swg, pair_tok, pair_w, moffsets,
                                            tilemap, d_ntiles, hg, nullptr, mffn, H, max_tiles,
                                            /*a_indirect=*/true, /*c_scatter=*/false, st);
            kernels::launch_pfm_moe_gemm_i8(mA_i8, msx, Wu_i8, swu, pair_tok, pair_w, moffsets,
                                            tilemap, d_ntiles, hu, nullptr, mffn, H, max_tiles,
                                            true, false, st);
            kernels::launch_prefill_swiglu(hg, hu, hh, (long)P * mffn, st);
            kernels::launch_prefill_quantize_rows_i8(hh, h_i8, sh, P, mffn, st);
            pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), st), "routed zero");
            kernels::launch_pfm_moe_gemm_i8(h_i8, sh, Wd_i8, swd, pair_tok, pair_w, moffsets,
                                            tilemap, d_ntiles, nullptr, routed_f32, H, mffn, max_tiles,
                                            /*a_indirect=*/false, /*c_scatter=*/true, st);
            // Shared expert (Qwen3.6 UD): out scaled by sigmoid(hn . gate_inp) per token.
            bf16* shared_out = nullptr;
            const void* sg = w.shared_gate_q ? w.shared_gate_q : w.shared_gate;
            if (c.n_shared > 0 && sg) {
                const int sgt = w.shared_gate_q ? w.shared_gate_qtype : 0;
                const void* su = w.shared_up_q ? w.shared_up_q : w.shared_up;
                const int sut = w.shared_up_q ? w.shared_up_qtype : 0;
                const void* sd = w.shared_down_q ? w.shared_down_q : w.shared_down;
                const int sdt = w.shared_down_q ? w.shared_down_qtype : 0;
                const bool has_gi = w.shared_gate_inp != nullptr;
                if (has_gi) {
                    const void* gi = w.shared_gate_inp_type
                        ? dq(w.shared_gate_inp, w.shared_gate_inp_type, 1, H) : w.shared_gate_inp;
                    kernels::launch_pfm_shared_gate(hn, gi, dw, N, H, st);
                }
                proj(hn, sg, sgt, sfg, mffn, H);
                proj(hn, su, sut, sfu, mffn, H);
                kernels::launch_pfm_shared_swiglu(sfg, sfu, has_gi ? dw : nullptr, sfh, N, mffn, st);
                proj(sfh, sd, sdt, ao, H, mffn);
                shared_out = ao;
            }
            // x = x + routed + shared (fp32 math); x already holds the post-attn residual, so this
            // fused add writes the final layer output directly (no separate ffn-out residual add).
            kernels::launch_pfm_resid3(x, routed_f32, shared_out, x, (long)N * H, st);
        }

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
    aw.free_all();
    return seed;
}

} // namespace sparkinfer
