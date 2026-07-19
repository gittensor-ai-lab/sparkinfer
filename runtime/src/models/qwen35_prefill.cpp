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
#include "sparkinfer/kernels/moe_prefill.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/quant.h"

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
    // 256/64 attn dims); they differ ONLY in the FFN, branched below (dense SwiGLU vs grouped MoE).
    const bool moe = !c.dense_ffn && c.n_experts > 0;
    if (!s.gguf || !c.hybrid || n <= 0) return -1;
    if (!c.dense_ffn && !moe) return -1;
    if (c.head_dim != 256 || c.linear_head_dim != 128) return -1;   // kernels specialize these
    if (moe && c.n_experts != 256) return -1;                       // router top-k path specialized for 256

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
    // Dense: int8 projections default ON. MoE: default OFF — the discrete top-k router amplifies the
    // per-token int8 projection error into different expert selections, which diverges from the
    // token-by-token path far more than in the dense FFN; bf16 projections keep the batched prefill
    // faithful to the decode path. SPARKINFER_PREFILL_I8 overrides either way.
    const char* _pi8 = getenv("SPARKINFER_PREFILL_I8");
    // Dense: int8 projections default ON. MoE: default OFF — the discrete top-k router amplifies the
    // per-token int8 projection error into different expert selections, which diverges from the
    // token-by-token path far more than in the dense FFN; bf16 projections keep the batched MoE
    // prefill faithful to the decode path. SPARKINFER_PREFILL_I8 overrides either way.
    bool use_i8 = _pi8 ? (_pi8[0] != '0') : !moe;
    // Long-context fidelity (dense): the near-1-decay GDN recurrence amplifies the per-row int8
    // activation-quant error across the sequence, so int8 prefill diverges from the token-by-token
    // path past ~96k (128k: top1 0.31 / KL 0.18). Above bf16_minctx (default 96k) fall back to bf16
    // projections, which stay faithful (128k: top1 0.69 / KL 0.04) at ~half the prefill throughput.
    // SPARKINFER_PREFILL_BF16_MINCTX overrides the threshold.
    static int bf16_minctx = []{ const char* e = getenv("SPARKINFER_PREFILL_BF16_MINCTX"); return e ? atoi(e) : 98304; }();
    if (N > bf16_minctx) use_i8 = false;
    Arena a8;
    // A_i8 holds the quantized activation. Dense: non-FFN projs quantize N rows x K(<=H); the chunked
    // FFN quantizes at most FC rows x ffn -> max of the two. MoE: no chunked FFN; the projections
    // (attn Q/K/V/O, GDN, shared) quantize N rows x maxAK (=max input dim, 4096 for Qwen3.6).
    const size_t a_i8_sz = moe ? (size_t)N * maxAK
                               : (((size_t)N * H > (size_t)FC * ffn) ? (size_t)N * H : (size_t)FC * ffn);
    signed char* A_i8 = use_i8 ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    signed char* W_i8 = use_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = use_i8 ? a8.alloc<float>((size_t)N) : nullptr;
    float* sw = use_i8 ? a8.alloc<float>((size_t)maxNO) : nullptr;
    if (use_i8 && !a8.ok) { a8.free_all(); use_i8 = false; }

    // ---- MoE (Qwen3.6) scratch: grouped expert FFN + shared expert ----
    const int E = moe ? c.n_experts : 0, tk = moe ? c.top_k : 0, mffn = moe ? c.moe_ffn : 0;
    const size_t P = (size_t)N * (tk > 0 ? tk : 1);
    bf16*  gate_bf = moe ? a.alloc<bf16>((size_t)E * mffn * H) : nullptr;   // dequant'd experts (once/layer)
    bf16*  up_bf   = moe ? a.alloc<bf16>((size_t)E * mffn * H) : nullptr;
    bf16*  down_bf = moe ? a.alloc<bf16>((size_t)E * H * mffn) : nullptr;
    float* rlog    = moe ? a.alloc<float>((size_t)N * (E ? E : 1)) : nullptr;
    int*   mf_ids  = moe ? a.alloc<int>(P) : nullptr;
    float* mf_wts  = moe ? a.alloc<float>(P) : nullptr;
    int*   mcounts = moe ? a.alloc<int>(E ? E : 1) : nullptr;
    int*   moff    = moe ? a.alloc<int>((E ? E : 1) + 1) : nullptr;
    int*   psrc    = moe ? a.alloc<int>(P) : nullptr;
    float* pw      = moe ? a.alloc<float>(P) : nullptr;
    bf16*  xperm   = moe ? a.alloc<bf16>(P * H) : nullptr;
    bf16*  gperm   = moe ? a.alloc<bf16>(P * (mffn ? mffn : 1)) : nullptr;
    bf16*  uperm   = moe ? a.alloc<bf16>(P * (mffn ? mffn : 1)) : nullptr;
    bf16*  hperm   = moe ? a.alloc<bf16>(P * (mffn ? mffn : 1)) : nullptr;
    bf16*  yperm   = moe ? a.alloc<bf16>(P * H) : nullptr;
    float* routf   = moe ? a.alloc<float>((size_t)N * H) : nullptr;
    bf16*  shg     = moe ? a.alloc<bf16>((size_t)N * (mffn ? mffn : 1)) : nullptr;
    bf16*  shu     = moe ? a.alloc<bf16>((size_t)N * (mffn ? mffn : 1)) : nullptr;
    bf16*  shh     = moe ? a.alloc<bf16>((size_t)N * (mffn ? mffn : 1)) : nullptr;
    bf16*  shout   = moe ? a.alloc<bf16>((size_t)N * H) : nullptr;
    float* dsw     = moe ? a.alloc<float>((size_t)N) : nullptr;
    const int moe_maxT = moe ? kernels::moe_prefill_grouped_maxtiles((int)P, E) : 0;
    int* sched_te = moe ? a.alloc<int>(moe_maxT) : nullptr;   // grouped-GEMM tile schedule (reused/layer)
    int* sched_tr = moe ? a.alloc<int>(moe_maxT) : nullptr;
    int* sched_dT = moe ? a.alloc<int>(1) : nullptr;
    if (moe && !a.ok) { a.free_all(); a8.free_all(); fprintf(stderr, "[prefill] MoE scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }

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

        if (!moe) {
            // dense SwiGLU FFN, chunked over tokens: ffg/ffu/A_i8 stay O(FC*ffn). Per-token
            // independent, so this is numerically identical to the full-width pass.
            for (int fo = 0; fo < N; fo += FC) {
                const int fn = (N - fo < FC) ? (N - fo) : FC;
                const bf16* hn_c = hn + (size_t)fo * H;
                proj(hn_c, w.gate_q, w.gate_qtype, ffg, ffn, H, fn);
                proj(hn_c, w.up_q,   w.up_qtype,   ffu, ffn, H, fn);
                kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                proj(ffg, w.down_q, w.down_qtype, ao + (size_t)fo * H, H, ffn, fn);
            }
        } else {
            // ---- grouped MoE FFN: weight-amortized over the N prompt tokens ----
            // router (fp32 logits, matching the per-token float router) -> top-8 + softmax + histogram
            kernels::launch_moe_prefill_router_logits(hn, dq(w.router_w, w.router_w_type, E, H), rlog, N, E, H, st);
            kernels::launch_moe_router(rlog, mf_ids, mf_wts, mcounts, N, E, tk, 1, st);
            // dequant all E experts once (Q4_K rows are independent -> one launch per tensor)
            kernels::launch_gguf_dequant(w.gate_qtype, w.gate_q, gate_bf, (long)E * mffn * H, st);
            kernels::launch_gguf_dequant(w.up_qtype,   w.up_q,   up_bf,   (long)E * mffn * H, st);
            kernels::launch_gguf_dequant(w.down_qtype, w.down_q, down_bf, (long)E * H * mffn, st);
            // permute tokens into per-expert groups; grouped GEMMs; swiglu; weighted scatter-add
            kernels::launch_moe_prefill_permute(mf_ids, mf_wts, mcounts, moff, psrc, pw, N, E, tk, st);
            const int Pn = N * tk;
            kernels::launch_moe_prefill_build_sched(moff, sched_te, sched_tr, sched_dT, E, st);
            kernels::launch_moe_prefill_gather(hn, psrc, xperm, Pn, H, st);
            kernels::launch_moe_prefill_grouped_gemm(xperm, gate_bf, moff, sched_te, sched_tr, sched_dT, gperm, Pn, E, mffn, H, st);
            kernels::launch_moe_prefill_grouped_gemm(xperm, up_bf,   moff, sched_te, sched_tr, sched_dT, uperm, Pn, E, mffn, H, st);
            kernels::launch_moe_prefill_swiglu(gperm, uperm, hperm, (long)Pn * mffn, st);
            kernels::launch_moe_prefill_grouped_gemm(hperm, down_bf, moff, sched_te, sched_tr, sched_dT, yperm, Pn, E, H, mffn, st);
            pf_cu(cudaMemsetAsync(routf, 0, (size_t)N * H * sizeof(float), st), "moe routed zero");
            kernels::launch_moe_prefill_scatter_weighted(yperm, psrc, pw, routf, Pn, H, st);
            // shared expert (dense, applied to every token) + optional scalar sigmoid gate
            const bf16* shared_ptr = nullptr; const float* dsw_ptr = nullptr;
            const void* sg = w.shared_gate_q ? w.shared_gate_q : w.shared_gate;
            if (c.n_shared > 0 && sg) {
                const void* su = w.shared_up_q   ? w.shared_up_q   : w.shared_up;
                const void* sd = w.shared_down_q ? w.shared_down_q : w.shared_down;
                const int sgt = w.shared_gate_q ? w.shared_gate_qtype : 0;
                const int sut = w.shared_up_q   ? w.shared_up_qtype   : 0;
                const int sdt = w.shared_down_q ? w.shared_down_qtype : 0;
                proj(hn, sg, sgt, shg, mffn, H);
                proj(hn, su, sut, shu, mffn, H);
                kernels::launch_prefill_swiglu(shg, shu, shh, (long)N * mffn, st);
                proj(shh, sd, sdt, shout, H, mffn);
                shared_ptr = shout;
                if (w.shared_gate_inp) {
                    kernels::launch_moe_shared_gate(hn, dq(w.shared_gate_inp, w.shared_gate_inp_type, 1, H), dsw, N, H, st);
                    dsw_ptr = dsw;
                }
            }
            kernels::launch_moe_prefill_finalize(routf, shared_ptr, dsw_ptr, ao, N, H, st);
        }

        // x += ffn_out (in-place residual) ; xn = RMSNorm(x, next_input_norm)  (final_norm on last layer)
        kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
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

} // namespace sparkinfer
