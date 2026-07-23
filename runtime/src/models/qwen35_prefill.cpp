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
#include "sparkinfer/kernels/prefill_fp8.h"
#include "sparkinfer/kernels/prefill_moe.h"
#include "sparkinfer/kernels/moe.h"

#include <cuda_runtime.h>
#include <algorithm>
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
    // MoE: optional int8 for shared-expert GEMMs only (attn/GDN/router stay bf16 — those feed
    // the top-k router). Distinct from full PREFILL_I8=1, #555 bf16 weight cache, and #566
    // live-expert coalesce/pair dequant. Env SPARKINFER_PREFILL_MOE_SHARED_I8=0 disables (A/B).
    bool moe_shared_i8 = moe && !use_i8 && [&]{
        const char* e = getenv("SPARKINFER_PREFILL_MOE_SHARED_I8");
        if (e) return e[0] == '1';
        return true;
    }();
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
    // Dense short-ctx: GDN recurrence amplifies per-row int8 activation error at the H3
    // prefill_check size (Qwythos @512: top1 0.6875 < 0.80). Keep GDN on bf16 for N<=512 by
    // default; FFN/full-attn stay on int8. Above 512, int8 GDN remains (CB @8k needs it).
    // SPARKINFER_PREFILL_I8_GDN=1/0 forces on/off at every N (A/B).
    const char* _pi8gdn = getenv("SPARKINFER_PREFILL_I8_GDN");
    const bool use_i8_gdn = !moe && use_i8 && [&]{
        if (_pi8gdn && _pi8gdn[0] == '1') return true;
        if (_pi8gdn && _pi8gdn[0] == '0') return false;
        return N > 512;
    }();
    // GDN projections (wqkv/wqkv_gate/ssm_out) at long ctx: run them on the fp8 (e4m3) tensor cores
    // instead of bf16. int8 is off here because the near-1-decay recurrence amplifies per-row int8
    // activation-quant error (128k top1 ~0.31); e4m3's floating range holds it to bf16-like fidelity
    // (~0.69) at the full int8 rate. The int8 activation scratch (A_i8/W_i8, 1 byte) doubles as the
    // e4m3 buffer -- fp8 GDN and int8 FFN/attn never run at the same instant within a layer.
    // SPARKINFER_PREFILL_FP8_GDN=0 restores the bf16 GDN projections (A/B).
    const char* _pfp8 = getenv("SPARKINFER_PREFILL_FP8_GDN");
    bool use_fp8_gdn = long_bf16 && (!_pfp8 || _pfp8[0] != '0');
    // MoE (Qwen3.6): optional fp8 (e4m3) attn/GDN projections. Default OFF — opt-in via
    // SPARKINFER_PREFILL_MOE_FP8=1. Kept off until prefill_check fidelity is clean on main
    // (see #586); the GPU MoE tilemap bug (#583) already corrupts batched prefill, and FP8
    // adds further TOP1 loss on top. Dense long-ctx GDN fp8 (PREFILL_FP8_GDN) is unaffected.
    const char* _pmfp8 = getenv("SPARKINFER_PREFILL_MOE_FP8");
    bool moe_fp8 = moe && _pmfp8 && _pmfp8[0] == '1';
    Arena a8;
    // A_i8 holds the quantized activation. Dense full-i8: non-FFN projs quantize N rows x K(<=H);
    // chunked FFN quantizes at most FC rows x ffn. Long-ctx selective: N*H if attn-i8/fp8-gdn else FC*ffn.
    // MoE: no chunked FFN; projections quantize N rows x maxAK.
    const bool wide_a = use_i8 || use_i8_attn || use_fp8_gdn || moe_fp8;
    const bool need_i8 = use_i8 || use_i8_ffn || use_i8_attn || use_fp8_gdn || moe_shared_i8 || moe_fp8;
    const size_t a_i8_sz = moe ? (size_t)N * maxAK
                               : (wide_a
                                  ? (((size_t)N * H > (size_t)FC * ffn) ? (size_t)N * H : (size_t)FC * ffn)
                                  : (size_t)FC * ffn);
    const size_t sx_n = (wide_a || moe_shared_i8) ? (size_t)N : (size_t)FC;
    signed char* A_i8 = need_i8 ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    signed char* W_i8 = need_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = need_i8 ? a8.alloc<float>(sx_n) : nullptr;
    float* sw = need_i8 ? a8.alloc<float>((size_t)maxNO) : nullptr;
    if (need_i8 && !a8.ok) {
        a8.free_all();
        use_i8 = false;
        use_i8_ffn = false;
        use_i8_attn = false;
        moe_shared_i8 = false;
        use_fp8_gdn = false;
        moe_fp8 = false;
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
    //
    const int E = moe ? c.n_experts : 0, topk = moe ? c.top_k : 0, mffn = moe ? c.moe_ffn : 0;
    const int P = moe ? N * topk : 0;                          // routed (token, expert) pairs
    // Short-N: BM=16 fills the tile (avg pairs/expert = N*8/256 = N/32; at 512 → 16).
    // Long-N: BM=128. Override with SPARKINFER_PREFILL_MOE_BM={16,128}.
    const int moe_bm = [&]{
        if (!moe) return 128;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_BM");
        if (e) { int v = atoi(e); return (v == 16) ? 16 : 128; }
        return (N <= 512) ? 16 : 128;
    }();
    const int max_tiles = moe ? (P + moe_bm - 1) / moe_bm + E : 0;
    // Opt-in fused QK path (experimental; currently slower than int8 materialize).
    const bool moe_fused = [&]{
        if (!moe) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_FUSED");
        return e && e[0] == '1';
    }();
    // Fused gate+up GEMM (BM=16): one A staging pass per K-tile. Default ON at short-N.
    const bool moe_fuse_gu = [&]{
        if (!moe || moe_bm != 16) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_FUSE_GU");
        return !e || e[0] != '0';
    }();
    // Expert-group L2 path: dequant G experts (~G*3 MB) then GEMM that group while hot in
    // L2. Wins at short-N (N<=512, G=32 → +5% @512); bulk is faster once tiles fill. Default
    // ON for N<=512. SPARKINFER_PREFILL_MOE_SERIAL=0/1 overrides; GROUP default 32.
    const bool moe_serial = [&]{
        if (!moe || moe_fused) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_SERIAL");
        if (e) return e[0] != '0';
        return N <= 512;
    }();
    // Device tilemap + mask dequant: skip per-layer D2H counts sync. Default OFF — opt-in via
    // SPARKINFER_PREFILL_MOE_GPU=1. The #583 default-ON path fails prefill_check (batched vs
    // token-loop TOP1 ~0.44–0.56 @512 vs ~0.88–0.94 with host tilemap; #586). Stale global
    // tilemap slots past the live group count can run GEMMs for wrong experts. Re-enable only
    // after tilemap invalidate + e<0 GEMM guards land and prefill_check passes.
    const bool moe_gpu = [&]{
        if (!moe_serial) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_GPU");
        return e && e[0] == '1';
    }();
    const int moe_group = [&]{
        const char* e = getenv("SPARKINFER_PREFILL_MOE_GROUP");
        int g = e ? atoi(e) : 32;
        if (g < 1) g = 1;
        if (g > 64) g = 64;
        return g;
    }();
    // Optional dual-stream weight ping-pong (env). Default OFF.
    const bool moe_pipe = [&]{
        if (!moe_serial) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_PIPE");
        return e && e[0] == '1';
    }();
    const int moe_slots = (moe_serial && moe_pipe) ? 2 : 1;
    // Opt-in MoE dequant overlap on stream_k/v. Default OFF: side-stream ops that write
    // ≳8KB permanently regress subsequent decode (~0.92×). SPARKINFER_PREFILL_MOE_OVERLAP=1
    // for experiments only.
    const bool moe_overlap = [&]{
        if (!moe_serial || !s.stream_k || !s.stream_v) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_OVERLAP");
        return e && e[0] == '1';
    }();
    // Hide shared-gate scalar behind MoE on stream_k. Default OFF: even a tiny side-stream
    // write + WaitEvent onto s.stream permanently slows decode graph replay (~0.92×).
    // SPARKINFER_PREFILL_HIDE_SG=1 for experiments only.
    const bool moe_hide_sg = [&]{
        if (!moe_serial || moe_overlap || !s.stream_k) return false;
        const char* e = getenv("SPARKINFER_PREFILL_HIDE_SG");
        return e && e[0] == '1';
    }();
    Arena am;
    signed char *Wg_i8 = nullptr, *Wu_i8 = nullptr, *Wd_i8 = nullptr, *h_i8 = nullptr, *mA_i8 = nullptr;
    float *swg = nullptr, *swu = nullptr, *swd = nullptr, *sh = nullptr, *msx = nullptr;
    float *mlogits = nullptr, *mweights = nullptr, *pair_w = nullptr, *routed_f32 = nullptr, *dw = nullptr;
    int *mids = nullptr, *mcounts = nullptr, *moffsets = nullptr, *mcursors = nullptr;
    int *pair_tok = nullptr, *tilemap = nullptr, *d_ntiles = nullptr, *d_live_le = nullptr;
    bf16 *hg = nullptr, *hu = nullptr, *hh = nullptr, *sfg = nullptr, *sfu = nullptr, *sfh = nullptr;
    if (moe) {
        if (!moe_fused) {
            // Serial: moe_slots * moe_group experts (ping-pong when piped). Bulk: full E.
            const int ew = moe_serial ? (moe_slots * moe_group) : E;
            Wg_i8 = am.alloc<signed char>((size_t)ew * mffn * H);
            Wu_i8 = am.alloc<signed char>((size_t)ew * mffn * H);
            Wd_i8 = am.alloc<signed char>((size_t)ew * H * mffn);
            swg = am.alloc<float>((size_t)ew * mffn);
            swu = am.alloc<float>((size_t)ew * mffn);
            swd = am.alloc<float>((size_t)ew * H);
        }
        mlogits = am.alloc<float>((size_t)N * E);
        mids = am.alloc<int>((size_t)P);
        mweights = am.alloc<float>((size_t)P);
        mcounts = am.alloc<int>(E);
        moffsets = am.alloc<int>(E + 1);
        mcursors = am.alloc<int>(E);
        pair_tok = am.alloc<int>((size_t)P);
        pair_w = am.alloc<float>((size_t)P);
        // Serial: packed tilemaps for all groups live in this buffer ([tm...][ntiles...]).
        tilemap = am.alloc<int>((size_t)2 * 2 * max_tiles);
        d_ntiles = am.alloc<int>(2);
        if (moe_serial) d_live_le = am.alloc<int>(E > 0 ? E : 256);
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
    // int8 activation memo: consecutive int8 projections of the SAME input (wq/wk/wv on xn,
    // wqkv/wqkv_gate on xn, FFN gate/up on the same chunk) re-quantize identical values into
    // A_i8/sx each call. Remember what A_i8 currently holds and skip the repeat quantize --
    // bit-identical, it reuses the exact bytes the first call produced. The memo is reset at
    // every layer top (xn/hn refresh in place) and wherever A_i8 is written outside proj().
    const bf16* a_q = nullptr; int a_qR = 0, a_qK = 0;
    auto quant_a_i8 = [&](const bf16* A, int R, int K) {
        if (a_q == A && a_qR == R && a_qK == K) return;
        kernels::launch_prefill_quantize_rows_i8(A, A_i8, sx, R, K, st);
        a_q = A; a_qR = R; a_qK = K;
    };
    // C[N,n_out] = A[N,K] @ W^T  (W native quantized [n_out,K]).
    auto proj = [&](const bf16* A, const void* W, int wtype, bf16* C, int n_out, int K, int rows = 0) {
        const int R = rows > 0 ? rows : N;   // rows (M) to process; chunked FFN passes a sub-N count
        // int8 only for the big weight-bound projections; keep the tiny per-v-head gate
        // projections (ssm_alpha/ssm_beta, n_out == v_heads) in bf16 — they feed the GDN
        // sigmoid gates, where per-row int8 quant of a 32-wide weight costs more accuracy
        // than the negligible time it saves.
        if (use_i8 && n_out >= 128) {
            quant_a_i8(A, R, K);
            // fused Q4_K/Q6_K -> int8 rows skips the dequant-to-bf16 scratch round trip
            if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
                const void* wb = dq(W, wtype, n_out, K);
                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
            }
            kernels::launch_prefill_gemm_i8(A_i8, W_i8, sx, sw, C, R, n_out, K, st);
        } else if ((use_fp8_gdn || moe_fp8) && n_out >= 128) {
            // fp8 (e4m3) tensor-core path for the long-ctx GDN projections. A_i8/W_i8 (1 byte) hold
            // the e4m3 operands; dequant the weight to bf16 scratch, then row/channel fp8-quantize.
            a_q = nullptr;                              // A_i8 becomes e4m3 -- invalidate the memo
            kernels::launch_prefill_quantize_rows_fp8(A, A_i8, sx, R, K, st);
            const void* wb = dq(W, wtype, n_out, K);
            kernels::launch_prefill_quantize_rows_fp8(wb, W_i8, sw, n_out, K, st);
            kernels::launch_prefill_gemm_fp8(A_i8, W_i8, sx, sw, C, R, n_out, K, st);
        } else {
            // mma.sync bf16 GEMM only for dense-hybrid long prefill (the >96k int8→bf16 fallback).
            // MoE reaches here only for the tiny n_out<128 gate projections or with the fp8 path
            // disabled; it stays on wmma (not mma.sync) in that fallback.
            // Gate on full prompt length N (not chunk rows R): FFN is token-chunked to FC=32k for
            // VRAM, so R<=FC would otherwise keep the dominant gate/up/down GEMMs on wmma forever.
            const bool prefer_mma = !moe && N > bf16_minctx;
            kernels::launch_prefill_gemm(A, dq(W, wtype, n_out, K), C, R, n_out, K, st, prefer_mma);
        }
    };

    // Residual-fused output projection: when the projection takes the int8 tensor-core path, run
    // it with the residual-fused GEMM straight into x (C[m,n] += dequant, pf_add's rounding), so
    // the ao scratch write + full-tensor add pass disappear. Returns false when the int8 path
    // would not be taken -- the caller falls back to proj() + launch_prefill_add, unchanged.
    // Bit-identical to the two-step path. SPARKINFER_PREFILL_RESID_FUSE=0 disables (A/B).
    const char* _prfuse = getenv("SPARKINFER_PREFILL_RESID_FUSE");
    const bool resid_fuse = !_prfuse || _prfuse[0] != '0';
    auto proj_resid = [&](const bf16* A, const void* W, int wtype, bf16* Cx, int n_out, int K,
                          int rows = 0) -> bool {
        if (!resid_fuse || !use_i8 || n_out < 128) return false;
        const int R = rows > 0 ? rows : N;
        quant_a_i8(A, R, K);
        if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
            const void* wb = dq(W, wtype, n_out, K);
            kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
        }
        kernels::launch_prefill_gemm_i8_resid(A_i8, W_i8, sx, sw, Cx, R, n_out, K, st);
        return true;
    };

    // GDN wqkv + wqkv_gate both project the same input xn, so on the fp8 path quantize xn to e4m3
    // ONCE and share it across both GEMMs (proj() would otherwise re-quantize xn per projection --
    // a full redundant read of xn and rewrite of the e4m3 activation each layer). Bit-identical to
    // the two independent proj() calls. Default on with either fp8 projection path (dense >96k GDN
    // or MoE); SPARKINFER_PREFILL_FP8_GDN_SHAREQ=0 restores the per-projection quantize (A/B).
    const char* _pshareq = getenv("SPARKINFER_PREFILL_FP8_GDN_SHAREQ");
    const bool fp8_shareq = (use_fp8_gdn || moe_fp8) && (!_pshareq || _pshareq[0] != '0');
    auto gdn_qkv_z = [&](const bf16* A, const Qwen35LayerWeights& w) {
        if (fp8_shareq) {
            a_q = nullptr;                              // A_i8 becomes e4m3 -- invalidate the memo
            kernels::launch_prefill_quantize_rows_fp8(A, A_i8, sx, N, H, st);   // xn -> e4m3 once
            const void* wb = dq(w.wqkv, w.wqkv_type, lqkv, H);
            kernels::launch_prefill_quantize_rows_fp8(wb, W_i8, sw, lqkv, H, st);
            kernels::launch_prefill_gemm_fp8(A_i8, W_i8, sx, sw, b8, N, lqkv, H, st);
            wb = dq(w.wqkv_gate, w.wqkv_gate_type, lvdim, H);
            kernels::launch_prefill_quantize_rows_fp8(wb, W_i8, sw, lvdim, H, st);
            kernels::launch_prefill_gemm_fp8(A_i8, W_i8, sx, sw, lz, N, lvdim, H, st);
        } else {
            proj(A, w.wqkv,      w.wqkv_type,      b8, lqkv,  H);   // qkv
            proj(A, w.wqkv_gate, w.wqkv_gate_type, lz, lvdim, H);   // z gate
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

    // MoE aux events: overlap path and/or tiny shared-gate hide on stream_k.
    cudaEvent_t moe_ev_up{}, moe_ev_down0{}, moe_ev_ready{}, moe_ev_sg{};
    if (moe_overlap) {
        pf_cu(cudaEventCreateWithFlags(&moe_ev_up, cudaEventDisableTiming), "moe ev_up");
        pf_cu(cudaEventCreateWithFlags(&moe_ev_down0, cudaEventDisableTiming), "moe ev_down0");
        pf_cu(cudaEventCreateWithFlags(&moe_ev_ready, cudaEventDisableTiming), "moe ev_ready");
    }
    if (moe_hide_sg)
        pf_cu(cudaEventCreateWithFlags(&moe_ev_sg, cudaEventDisableTiming), "moe ev_sg");

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        a_q = nullptr;                                 // xn/hn are refreshed in place each layer
        bool attn_fused = false;                       // post-attn residual folded into the proj?
        if (w.linear_attn) {
            // ---- Gated DeltaNet linear-attention layer ----
            // Short-ctx dense: hold GDN on bf16 unless SPARKINFER_PREFILL_I8_GDN=1.
            const bool restore_i8_gdn = use_i8;
            if (use_i8 && !use_i8_gdn) use_i8 = false;
            gdn_qkv_z(xn, w);                                       // qkv + z gate (fp8: fused)
            proj(xn, w.ssm_alpha, w.ssm_alpha_type, la, vh,    H);
            proj(xn, w.ssm_beta,  w.ssm_beta_type,  lb, vh,    H);
            bf16* conv_state = lin_conv_state + (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
            kernels::launch_prefill_gdn_conv(b8, w.ssm_conv, conv_state, gq, gk, gv,
                N, c.linear_q_heads, vh, c.linear_head_dim, c.linear_conv_kernel, eps, st);
            float* layer_state = s.lin_state + (size_t)L * vh * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_prefill_gdn_scan(gq, gk, gv, la, lb, w.ssm_dt, w.ssm_a,
                layer_state, att, N, c.linear_q_heads, vh, c.linear_head_dim, st);
            kernels::launch_prefill_gated_norm(att, lz, w.ssm_norm, lnrm, N, vh, c.linear_head_dim, eps, st);
            attn_fused = proj_resid(lnrm, w.ssm_out, w.ssm_out_type, x, H, lvdim);
            if (!attn_fused) proj(lnrm, w.ssm_out, w.ssm_out_type, ao, H, lvdim);
            use_i8 = restore_i8_gdn;
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
            attn_fused = proj_resid(att, w.wo, w.wo_type, x, H, qdim);
            if (!attn_fused) proj(att, w.wo, w.wo_type, ao, H, qdim);
            use_i8 = restore_i8;
        }

        // x += ao (post-attn residual, in-place; skipped when folded into the output proj)
        // hn = RMSNorm(x, post_attn_norm)
        if (!attn_fused) kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
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
            // The down projection takes the int8 path on both branches whenever ffn_i8 or use_i8,
            // so the FFN residual can ride the residual-fused GEMM straight into x per chunk.
            const bool ffn_fused = resid_fuse && (ffn_i8 || use_i8);
            for (int fo = 0; fo < N; fo += FC) {
                const int fn = (N - fo < FC) ? (N - fo) : FC;
                const bf16* hn_c = hn + (size_t)fo * H;
                if (ffn_i8) {
                    a_q = nullptr;                     // this branch writes A_i8/sx directly
                    kernels::launch_prefill_quantize_rows_i8(hn_c, A_i8, sx, fn, H, st);
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wg_i8, sx, ffn_swg, ffg, fn, ffn, H, st);
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wu_i8, sx, ffn_swu, ffu, fn, ffn, H, st);
                    // fused SwiGLU + int8 quantize for the down input (skips the ffg DRAM round-trip)
                    kernels::launch_prefill_swiglu_quant_i8(ffg, ffu, A_i8, sx, fn, ffn, st);
                    if (ffn_fused)
                        kernels::launch_prefill_gemm_i8_resid(A_i8, ffn_Wd_i8, sx, ffn_swd,
                                                              x + (size_t)fo * H, fn, H, ffn, st);
                    else
                        kernels::launch_prefill_gemm_i8(A_i8, ffn_Wd_i8, sx, ffn_swd,
                                                        ao + (size_t)fo * H, fn, H, ffn, st);
                } else {
                    proj(hn_c, w.gate_q, w.gate_qtype, ffg, ffn, H, fn);
                    proj(hn_c, w.up_q,   w.up_qtype,   ffu, ffn, H, fn);
                    if (use_i8) {
                        // Same fused SwiGLU + per-row int8 quantize the long-ctx ffn_i8 branch
                        // runs (bit-identical to swiglu-then-quantize; both bf16-round first) --
                        // skips the ffg store + reload that proj()'s internal quantize would pay.
                        a_q = nullptr;                 // swiglu_quant writes A_i8/sx directly
                        kernels::launch_prefill_swiglu_quant_i8(ffg, ffu, A_i8, sx, fn, ffn, st);
                        if (!kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q,
                                                                  W_i8, sw, H, ffn, st)) {
                            const void* wb = dq(w.down_q, w.down_qtype, H, ffn);
                            kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, H, ffn, st);
                        }
                        if (ffn_fused)
                            kernels::launch_prefill_gemm_i8_resid(A_i8, W_i8, sx, sw,
                                                                  x + (size_t)fo * H, fn, H, ffn, st);
                        else
                            kernels::launch_prefill_gemm_i8(A_i8, W_i8, sx, sw,
                                                            ao + (size_t)fo * H, fn, H, ffn, st);
                    } else {
                        kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                        if (!ffn_fused || !proj_resid(ffg, w.down_q, w.down_qtype,
                                                      x + (size_t)fo * H, H, ffn, fn))
                            proj(ffg, w.down_q, w.down_qtype, ao + (size_t)fo * H, H, ffn, fn);
                    }
                }
            }
            // x += ffn_out (skipped when the down GEMM already accumulated into x per chunk)
            if (!ffn_fused) kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
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
            kernels::launch_pfm_bucket_pairs_bm(mids, mweights, mcounts, moffsets, mcursors,
                                                pair_tok, pair_w, tilemap, d_ntiles, N, E, topk, moe_bm, st);
            kernels::launch_prefill_quantize_rows_i8(hn, mA_i8, msx, N, H, st);
            bool sg_hid = false;  // shared-gate scalar already on stream_k
            if (moe_fused) {
                // On-the-fly Q→bf16 B staging — no full-expert int8 materialize (experimental).
                kernels::launch_pfm_moe_gemm_qk(mA_i8, msx, w.gate_q, w.gate_qtype, pair_tok, pair_w,
                                                moffsets, tilemap, d_ntiles, hg, nullptr, mffn, H, max_tiles,
                                                /*a_indirect=*/true, /*c_scatter=*/false, st);
                kernels::launch_pfm_moe_gemm_qk(mA_i8, msx, w.up_q, w.up_qtype, pair_tok, pair_w,
                                                moffsets, tilemap, d_ntiles, hu, nullptr, mffn, H, max_tiles,
                                                true, false, st);
                kernels::launch_prefill_swiglu(hg, hu, hh, (long)P * mffn, st);
                kernels::launch_prefill_quantize_rows_i8(hh, h_i8, sh, P, mffn, st);
                pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), st), "routed zero");
                kernels::launch_pfm_moe_gemm_qk(h_i8, sh, w.down_q, w.down_qtype, pair_tok, pair_w,
                                                moffsets, tilemap, d_ntiles, nullptr, routed_f32, H, mffn, max_tiles,
                                                /*a_indirect=*/false, /*c_scatter=*/true, st);
            } else if (moe_serial) {
                // Expert-group L2 path on top of main(#561), tuned for N≈512:
                //   - D2H counts, skip empty groups, exact-ntm GEMM grids
                //   - coalesce live-expert dequant runs (skip empty weight rows)
                //   - one-shot packed tilemap H2D
                //   - optional gate∥up on private streams (decode-unsafe; env)
                auto q_row_bytes = [](int qtype, int cols) -> size_t {
                    const int bs = (qtype == 12) ? 144 : (qtype == 13) ? 176 : 210;
                    return (size_t)(cols >> 8) * (size_t)bs;
                };
                const size_t g_rb = q_row_bytes(w.gate_qtype, H);
                const size_t u_rb = q_row_bytes(w.up_qtype, H);
                const size_t d_rb = q_row_bytes(w.down_qtype, mffn);
                const size_t g_eb = (size_t)mffn * g_rb;
                const size_t u_eb = (size_t)mffn * u_rb;
                const size_t d_eb = (size_t)H * d_rb;
                const int G = moe_group;
                pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), st),
                      "routed zero");

                if (moe_gpu && !moe_overlap) {
                    // No D2H counts sync: device tilemap + mask dequant per expert group.
                    for (int base = 0; base < E; base += G) {
                        const int n_in = (E - base < G) ? (E - base) : G;
                        kernels::launch_pfm_group_tilemap(
                            mcounts, tilemap, d_ntiles, base, n_in, moe_bm, max_tiles, st);
                        const void* ge0 = (const char*)w.gate_q + (size_t)base * g_eb;
                        const void* ue0 = (const char*)w.up_q + (size_t)base * u_eb;
                        if (w.gate_qtype == w.up_qtype) {
                            kernels::launch_gguf_dequant_rows_i8_mask_pair(
                                w.gate_qtype, ge0, Wg_i8, swg, ue0, Wu_i8, swu,
                                mcounts, base, n_in, mffn, H, g_eb, u_eb, st);
                        } else {
                            kernels::launch_gguf_dequant_rows_i8_mask(
                                w.gate_qtype, ge0, Wg_i8, swg, mcounts, base, n_in, mffn, H,
                                g_eb, st);
                            kernels::launch_gguf_dequant_rows_i8_mask(
                                w.up_qtype, ue0, Wu_i8, swu, mcounts, base, n_in, mffn, H,
                                u_eb, st);
                        }
                        if (moe_fuse_gu) {
                            kernels::launch_pfm_moe_gemm_i8_gate_up_bm16(
                                mA_i8, msx, Wg_i8, swg, Wu_i8, swu, pair_tok, moffsets, tilemap,
                                d_ntiles, hg, hu, mffn, H, max_tiles, base, st);
                        } else {
                            kernels::launch_pfm_moe_gemm_i8_bm_base(
                                mA_i8, msx, Wg_i8, swg, pair_tok, pair_w, moffsets, tilemap,
                                d_ntiles, hg, nullptr, mffn, H, max_tiles, moe_bm, base,
                                true, false, st);
                            kernels::launch_pfm_moe_gemm_i8_bm_base(
                                mA_i8, msx, Wu_i8, swu, pair_tok, pair_w, moffsets, tilemap,
                                d_ntiles, hu, nullptr, mffn, H, max_tiles, moe_bm, base,
                                true, false, st);
                        }
                    }
                    kernels::launch_prefill_swiglu(hg, hu, hh, (long)P * mffn, st);
                    kernels::launch_prefill_quantize_rows_i8(hh, h_i8, sh, P, mffn, st);
                    for (int base = 0; base < E; base += G) {
                        const int n_in = (E - base < G) ? (E - base) : G;
                        kernels::launch_pfm_group_tilemap(
                            mcounts, tilemap, d_ntiles, base, n_in, moe_bm, max_tiles, st);
                        const void* de0 = (const char*)w.down_q + (size_t)base * d_eb;
                        kernels::launch_gguf_dequant_rows_i8_mask(
                            w.down_qtype, de0, Wd_i8, swd, mcounts, base, n_in, H, mffn, d_eb, st);
                        kernels::launch_pfm_moe_gemm_i8_bm_base(
                            h_i8, sh, Wd_i8, swd, pair_tok, pair_w, moffsets, tilemap, d_ntiles,
                            nullptr, routed_f32, H, mffn, max_tiles, moe_bm, base,
                            false, true, st);
                    }
                } else {
                int h_counts_stack[256];
                int* h_counts = h_counts_stack;
                // Pinned counts make the D2H sync cheaper (pageable stalls the GPU).
                static thread_local int* pinned_counts = nullptr;
                if (!pinned_counts) {
                    if (cudaMallocHost(&pinned_counts, 256 * sizeof(int)) != cudaSuccess)
                        pinned_counts = nullptr;
                }
                if (pinned_counts) h_counts = pinned_counts;
                pf_cu(cudaMemcpyAsync(h_counts, mcounts, (size_t)E * sizeof(int),
                                      cudaMemcpyDeviceToHost, st), "moe counts D2H");
                pf_cu(cudaStreamSynchronize(st), "moe serial sync");

                struct ActiveGroup {
                    int base, n_in, ntm, tm_off, n_live, live_off;
                    std::vector<int> tm;
                    std::vector<int> live;
                };
                static thread_local std::vector<ActiveGroup> active;
                static thread_local std::vector<int> h_tm_all, h_nt_all, h_live_all;
                active.clear();
                active.reserve((size_t)((E + G - 1) / G));
                int tm_total = 0;
                int live_total = 0;
                for (int base = 0; base < E; base += G) {
                    const int n_in = (E - base < G) ? (E - base) : G;
                    ActiveGroup ag;
                    ag.base = base;
                    ag.n_in = n_in;
                    ag.ntm = 0;
                    ag.tm_off = tm_total;
                    ag.n_live = 0;
                    ag.live_off = live_total;
                    ag.tm.reserve((size_t)2 * 64);
                    ag.live.reserve((size_t)n_in);
                    for (int le = 0; le < n_in; le++) {
                        const int e = base + le;
                        const int cnt = h_counts[e];
                        if (cnt <= 0) continue;
                        ag.live.push_back(le);
                        ag.n_live++;
                        const int nt = (cnt + moe_bm - 1) / moe_bm;
                        for (int mt = 0; mt < nt; mt++) {
                            if (ag.ntm >= max_tiles) break;
                            ag.tm.push_back(e);
                            ag.tm.push_back(mt);
                            ag.ntm++;
                        }
                    }
                    if (ag.ntm > 0) {
                        tm_total += ag.ntm;
                        live_total += ag.n_live;
                        active.push_back(std::move(ag));
                    }
                }
                const int n_active = (int)active.size();

                h_tm_all.resize((size_t)2 * std::max(tm_total, 1));
                h_nt_all.resize((size_t)std::max(n_active, 1));
                h_live_all.resize((size_t)std::max(live_total, 1));
                for (int gi = 0; gi < n_active; gi++) {
                    const ActiveGroup& ag = active[(size_t)gi];
                    h_nt_all[(size_t)gi] = ag.ntm;
                    for (int t = 0; t < ag.ntm; t++) {
                        h_tm_all[(size_t)2 * (ag.tm_off + t)]     = ag.tm[(size_t)2 * t];
                        h_tm_all[(size_t)2 * (ag.tm_off + t) + 1] = ag.tm[(size_t)2 * t + 1];
                    }
                    for (int i = 0; i < ag.n_live; i++)
                        h_live_all[(size_t)ag.live_off + i] = ag.live[(size_t)i];
                }
                const bool pack_fit = (tm_total <= 2 * max_tiles) && (n_active <= 64) &&
                                     (2 * tm_total + n_active <= 4 * max_tiles);

                // Reuse decode stream_k (=sa) / stream_v (=sb) for MoE; host-join so s.stream
                // never accumulates cross-stream WaitEvents from the group loop.
                cudaStream_t sa = moe_overlap ? s.stream_k : st;
                cudaStream_t sb = moe_overlap ? s.stream_v : st;
                if (moe_overlap) {
                    pf_cu(cudaEventRecord(moe_ev_ready, st), "moe fork");
                    pf_cu(cudaStreamWaitEvent(sa, moe_ev_ready, 0), "moe sa wait");
                    pf_cu(cudaStreamWaitEvent(sb, moe_ev_ready, 0), "moe sb wait");
                    pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), sa),
                          "routed zero");
                }

                int* d_nt_pack = nullptr;
                if (pack_fit && n_active > 0) {
                    d_nt_pack = tilemap + 2 * tm_total;
                    pf_cu(cudaMemcpyAsync(tilemap, h_tm_all.data(),
                                          (size_t)2 * tm_total * sizeof(int),
                                          cudaMemcpyHostToDevice, sa), "moe all tm H2D");
                    pf_cu(cudaMemcpyAsync(d_nt_pack, h_nt_all.data(),
                                          (size_t)n_active * sizeof(int),
                                          cudaMemcpyHostToDevice, sa), "moe all nt H2D");
                }
                // One-shot live-expert index upload (stable for all group gathers — no per-group race).
                if (d_live_le && live_total > 0) {
                    pf_cu(cudaMemcpyAsync(d_live_le, h_live_all.data(),
                                          (size_t)live_total * sizeof(int),
                                          cudaMemcpyHostToDevice, sa), "moe all live H2D");
                }

                auto dq_gateup = [&](const ActiveGroup& ag) {
                    const int n_live = ag.n_live;
                    if (n_live <= 0) return;
                    const void* ge0 = (const char*)w.gate_q +
                        (size_t)ag.base * (size_t)mffn * g_rb;
                    const void* ue0 = (const char*)w.up_q +
                        (size_t)ag.base * (size_t)mffn * u_rb;
                    const size_t g_eb = (size_t)mffn * g_rb;
                    const size_t u_eb = (size_t)mffn * u_rb;
                    // Full group live + same type: one fused contiguous pair launch.
                    if (n_live == ag.n_in && !moe_overlap && w.gate_qtype == w.up_qtype) {
                        if (kernels::launch_gguf_dequant_rows_i8_pair(
                                w.gate_qtype, ge0, Wg_i8, swg, ue0, Wu_i8, swu,
                                ag.n_in * mffn, H, sa))
                            return;
                    }
                    if (n_live == ag.n_in && !moe_overlap) {
                        kernels::launch_gguf_dequant_rows_i8(
                            w.gate_qtype, ge0, Wg_i8, swg, ag.n_in * mffn, H, sa);
                        kernels::launch_gguf_dequant_rows_i8(
                            w.up_qtype, ue0, Wu_i8, swu, ag.n_in * mffn, H, sa);
                        return;
                    }
                    // Sparse: optional one-shot gather (SPARKINFER_PREFILL_MOE_GATHER=1).
                    // Default OFF: gather still needs accuracy bake-off; coalesce+pair is the win.
                    static const int use_gather = [] {
                        const char* e = getenv("SPARKINFER_PREFILL_MOE_GATHER");
                        return (e && e[0] == '1') ? 1 : 0;
                    }();
                    const int* d_live = d_live_le + ag.live_off;
                    const int* h_live = ag.live.data();
                    if (use_gather && !moe_overlap && w.gate_qtype == w.up_qtype && d_live_le &&
                        kernels::launch_gguf_dequant_rows_i8_gather_pair(
                            w.gate_qtype, ge0, Wg_i8, swg, ue0, Wu_i8, swu,
                            d_live, n_live, mffn, H, g_eb, u_eb, sa))
                        return;
                    // Pair each contiguous live run (same-type gate+up) — big @512 win vs serial.
                    if (!moe_overlap && w.gate_qtype == w.up_qtype) {
                        int i = 0;
                        while (i < n_live) {
                            int le0 = h_live[i], run = 1;
                            while (i + run < n_live && h_live[i + run] == le0 + run) run++;
                            const void* ge = (const char*)ge0 + (size_t)le0 * g_eb;
                            const void* ue = (const char*)ue0 + (size_t)le0 * u_eb;
                            if (!kernels::launch_gguf_dequant_rows_i8_pair(
                                    w.gate_qtype, ge, Wg_i8 + (size_t)le0 * mffn * H,
                                    swg + (size_t)le0 * mffn,
                                    ue, Wu_i8 + (size_t)le0 * mffn * H,
                                    swu + (size_t)le0 * mffn, run * mffn, H, sa)) {
                                kernels::launch_gguf_dequant_rows_i8(
                                    w.gate_qtype, ge, Wg_i8 + (size_t)le0 * mffn * H,
                                    swg + (size_t)le0 * mffn, run * mffn, H, sa);
                                kernels::launch_gguf_dequant_rows_i8(
                                    w.up_qtype, ue, Wu_i8 + (size_t)le0 * mffn * H,
                                    swu + (size_t)le0 * mffn, run * mffn, H, sa);
                            }
                            i += run;
                        }
                        return;
                    }
                    auto dq_g = [&](int qtype, const void* src0, signed char* dst, float* sc,
                                    size_t eb, cudaStream_t ds) {
                        if (use_gather && d_live_le &&
                            kernels::launch_gguf_dequant_rows_i8_gather(
                                qtype, src0, dst, sc, d_live, n_live, mffn, H, eb, ds))
                            return;
                        int i = 0;
                        while (i < n_live) {
                            int le0 = h_live[i], run = 1;
                            while (i + run < n_live && h_live[i + run] == le0 + run) run++;
                            const void* src = (const char*)src0 + (size_t)le0 * eb;
                            kernels::launch_gguf_dequant_rows_i8(
                                qtype, src, dst + (size_t)le0 * mffn * H,
                                sc + (size_t)le0 * mffn, run * mffn, H, ds);
                            i += run;
                        }
                    };
                    dq_g(w.gate_qtype, ge0, Wg_i8, swg, g_eb, sa);
                    if (moe_overlap) {
                        dq_g(w.up_qtype, ue0, Wu_i8, swu, u_eb, sb);
                        pf_cu(cudaEventRecord(moe_ev_up, sb), "moe up done");
                        pf_cu(cudaStreamWaitEvent(sa, moe_ev_up, 0), "moe sa wait up");
                    } else {
                        dq_g(w.up_qtype, ue0, Wu_i8, swu, u_eb, sa);
                    }
                };
                auto dq_down = [&](const ActiveGroup& ag, cudaStream_t ds) {
                    const int n_live = ag.n_live;
                    if (n_live <= 0) return;
                    const void* de0 = (const char*)w.down_q +
                        (size_t)ag.base * (size_t)H * d_rb;
                    const size_t d_eb = (size_t)H * d_rb;
                    if (n_live == ag.n_in) {
                        kernels::launch_gguf_dequant_rows_i8(
                            w.down_qtype, de0, Wd_i8, swd, ag.n_in * H, mffn, ds);
                        return;
                    }
                    const int* d_live = d_live_le + ag.live_off;
                    const int* h_live = ag.live.data();
                    static const int use_gather_dn = [] {
                        const char* e = getenv("SPARKINFER_PREFILL_MOE_GATHER");
                        return (e && e[0] == '1') ? 1 : 0;
                    }();
                    if (use_gather_dn && d_live_le &&
                        kernels::launch_gguf_dequant_rows_i8_gather(
                            w.down_qtype, de0, Wd_i8, swd, d_live, n_live, H, mffn, d_eb, ds))
                        return;
                    int i = 0;
                    while (i < n_live) {
                        int le0 = h_live[i], run = 1;
                        while (i + run < n_live && h_live[i + run] == le0 + run) run++;
                        const void* de = (const char*)de0 + (size_t)le0 * d_eb;
                        kernels::launch_gguf_dequant_rows_i8(
                            w.down_qtype, de, Wd_i8 + (size_t)le0 * H * mffn,
                            swd + (size_t)le0 * H, run * H, mffn, ds);
                        i += run;
                    }
                };

                // Kick tiny shared-gate scalar on stream_k (≈2KB write) behind MoE.
                // Only when gate_inp is already float — quantized dq() would trash MoE scratch.
                if (moe_hide_sg && c.n_shared > 0 && w.shared_gate_inp &&
                    !w.shared_gate_inp_type &&
                    (w.shared_gate_q || w.shared_gate)) {
                    kernels::launch_pfm_shared_gate(
                        hn, w.shared_gate_inp, dw, N, H, s.stream_k);
                    pf_cu(cudaEventRecord(moe_ev_sg, s.stream_k), "moe sg done");
                    sg_hid = true;
                }

                for (int gi = 0; gi < n_active; gi++) {
                    const ActiveGroup& ag = active[(size_t)gi];
                    int* tm;
                    int* nt;
                    if (pack_fit) {
                        tm = tilemap + 2 * ag.tm_off;
                        nt = d_nt_pack + gi;
                    } else {
                        tm = tilemap;
                        nt = d_ntiles;
                        pf_cu(cudaMemcpyAsync(tm, ag.tm.data(),
                                              (size_t)2 * ag.ntm * sizeof(int),
                                              cudaMemcpyHostToDevice, sa), "moe tm H2D");
                        pf_cu(cudaMemcpyAsync(nt, &ag.ntm, sizeof(int),
                                              cudaMemcpyHostToDevice, sa), "moe ntm H2D");
                    }
                    dq_gateup(ag);
                    if (moe_fuse_gu) {
                        kernels::launch_pfm_moe_gemm_i8_gate_up_bm16(
                            mA_i8, msx, Wg_i8, swg, Wu_i8, swu, pair_tok, moffsets, tm, nt,
                            hg, hu, mffn, H, ag.ntm, ag.base, sa);
                    } else {
                        kernels::launch_pfm_moe_gemm_i8_bm_base(
                            mA_i8, msx, Wg_i8, swg, pair_tok, pair_w, moffsets, tm, nt,
                            hg, nullptr, mffn, H, ag.ntm, moe_bm, ag.base,
                            /*a_indirect=*/true, /*c_scatter=*/false, sa);
                        kernels::launch_pfm_moe_gemm_i8_bm_base(
                            mA_i8, msx, Wu_i8, swu, pair_tok, pair_w, moffsets, tm, nt,
                            hu, nullptr, mffn, H, ag.ntm, moe_bm, ag.base, true, false, sa);
                    }
                }

                kernels::launch_prefill_swiglu(hg, hu, hh, (long)P * mffn, sa);
                kernels::launch_prefill_quantize_rows_i8(hh, h_i8, sh, P, mffn, sa);

                for (int gi = 0; gi < n_active; gi++) {
                    const ActiveGroup& ag = active[(size_t)gi];
                    int* tm;
                    int* nt;
                    if (pack_fit) {
                        tm = tilemap + 2 * ag.tm_off;
                        nt = d_nt_pack + gi;
                    } else {
                        tm = tilemap;
                        nt = d_ntiles;
                        pf_cu(cudaMemcpyAsync(tm, ag.tm.data(),
                                              (size_t)2 * ag.ntm * sizeof(int),
                                              cudaMemcpyHostToDevice, sa), "moe down tm");
                        pf_cu(cudaMemcpyAsync(nt, &ag.ntm, sizeof(int),
                                              cudaMemcpyHostToDevice, sa), "moe down ntm");
                    }
                    dq_down(ag, sa);
                    kernels::launch_pfm_moe_gemm_i8_bm_base(
                        h_i8, sh, Wd_i8, swd, pair_tok, pair_w, moffsets, tm, nt,
                        nullptr, routed_f32, H, mffn, ag.ntm, moe_bm, ag.base,
                        /*a_indirect=*/false, /*c_scatter=*/true, sa);
                }
                if (moe_overlap) {
                    // Host-join private streams — do NOT cudaStreamWaitEvent onto s.stream
                    // (cross-stream waits on the decode stream permanently slow graph replay).
                    pf_cu(cudaStreamSynchronize(sa), "moe sa join");
                }
                }  // !moe_gpu host tilemap path
            } else {
                // Bulk: expert weights -> int8 rows ONCE per layer (all 256 experts).
                kernels::launch_gguf_dequant_rows_i8(w.gate_qtype, w.gate_q, Wg_i8, swg, E * mffn, H, st);
                kernels::launch_gguf_dequant_rows_i8(w.up_qtype,   w.up_q,   Wu_i8, swu, E * mffn, H, st);
                kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q, Wd_i8, swd, E * H, mffn, st);
                if (moe_fuse_gu) {
                    kernels::launch_pfm_moe_gemm_i8_gate_up_bm16(
                        mA_i8, msx, Wg_i8, swg, Wu_i8, swu, pair_tok, moffsets, tilemap, d_ntiles,
                        hg, hu, mffn, H, max_tiles, /*e_base=*/0, st);
                } else {
                    kernels::launch_pfm_moe_gemm_i8_bm(mA_i8, msx, Wg_i8, swg, pair_tok, pair_w, moffsets,
                                                       tilemap, d_ntiles, hg, nullptr, mffn, H, max_tiles, moe_bm,
                                                       /*a_indirect=*/true, /*c_scatter=*/false, st);
                    kernels::launch_pfm_moe_gemm_i8_bm(mA_i8, msx, Wu_i8, swu, pair_tok, pair_w, moffsets,
                                                       tilemap, d_ntiles, hu, nullptr, mffn, H, max_tiles, moe_bm,
                                                       true, false, st);
                }
                kernels::launch_prefill_swiglu(hg, hu, hh, (long)P * mffn, st);
                kernels::launch_prefill_quantize_rows_i8(hh, h_i8, sh, P, mffn, st);
                pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), st), "routed zero");
                kernels::launch_pfm_moe_gemm_i8_bm(h_i8, sh, Wd_i8, swd, pair_tok, pair_w, moffsets,
                                                   tilemap, d_ntiles, nullptr, routed_f32, H, mffn, max_tiles, moe_bm,
                                                   /*a_indirect=*/false, /*c_scatter=*/true, st);
            }
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
                    if (sg_hid) {
                        // Join tiny stream_k write before dw is consumed on st.
                        pf_cu(cudaStreamWaitEvent(st, moe_ev_sg, 0), "moe wait sg");
                    } else {
                        const void* gi = w.shared_gate_inp_type
                            ? dq(w.shared_gate_inp, w.shared_gate_inp_type, 1, H)
                            : w.shared_gate_inp;
                        kernels::launch_pfm_shared_gate(hn, gi, dw, N, H, st);
                    }
                }
                const bool restore_i8_sh = use_i8;
                if (moe_shared_i8) use_i8 = true;
                proj(hn, sg, sgt, sfg, mffn, H);
                proj(hn, su, sut, sfu, mffn, H);
                kernels::launch_pfm_shared_swiglu(sfg, sfu, has_gi ? dw : nullptr, sfh, N, mffn, st);
                proj(sfh, sd, sdt, ao, H, mffn);
                use_i8 = restore_i8_sh;
                shared_out = ao;
            }
            // x = x + routed + shared (fp32 math); x already holds the post-attn residual, so this
            // fused add writes the final layer output directly (no separate ffn-out residual add).
            kernels::launch_pfm_resid3(x, routed_f32, shared_out, x, (long)N * H, st);
        }

        const void* next_norm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_rmsnorm(x, next_norm, xn, N, H, eps, st);
    }

    if (moe_overlap) {
        pf_cu(cudaStreamSynchronize(s.stream_k), "moe sk sync");
        pf_cu(cudaStreamSynchronize(s.stream_v), "moe sv sync");
        cudaEventDestroy(moe_ev_up);
        cudaEventDestroy(moe_ev_down0);
        cudaEventDestroy(moe_ev_ready);
    }
    if (moe_hide_sg)
        cudaEventDestroy(moe_ev_sg);

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
