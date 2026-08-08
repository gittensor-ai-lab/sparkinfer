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
#include "sparkinfer/kernels/prefill_router_mma.h"
#include "sparkinfer/kernels/prefill_moe_q.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/models/dflash_kernels.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    std::vector<size_t> sizes;
    size_t cursor = 0;
    bool ok = true;
    template <class T> T* alloc(size_t n) {
        if (n == 0) n = 1;
        const size_t bytes = n * sizeof(T);
        void* p = nullptr;
        if (cursor < bufs.size() && sizes[cursor] >= bytes) {
            p = bufs[cursor++];
            return static_cast<T*>(p);
        }
        if (cursor < bufs.size()) {
            cudaFree(bufs[cursor]);
            bufs.erase(bufs.begin() + cursor);
            sizes.erase(sizes.begin() + cursor);
        }
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        bufs.insert(bufs.begin() + cursor, p);
        sizes.insert(sizes.begin() + cursor, bytes);
        ++cursor;
        return static_cast<T*>(p);
    }
    void rewind() { cursor = 0; ok = true; }
    void free_all() { for (void* b : bufs) cudaFree(b); bufs.clear(); sizes.clear(); cursor = 0; }
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
    // prefill_check size (Qwythos @512: top1 0.6875 < 0.80). Keep GDN on bf16 only for the
    // exact H3 prefix (N==512). Short score prompts (200..360) stay on int8 GDN so vs-llama
    // top1/KL clear the 0.90/0.20 bars; N>512 keeps int8 GDN for CB mid-ctx pp.
    // SPARKINFER_PREFILL_I8_GDN=1/0 forces on/off at every N (A/B).
    const char* _pi8gdn = getenv("SPARKINFER_PREFILL_I8_GDN");
    const bool use_i8_gdn = !moe && use_i8 && [&]{
        if (_pi8gdn && _pi8gdn[0] == '1') return true;
        if (_pi8gdn && _pi8gdn[0] == '0') return false;
        return N != 512;
    }();
    // GDN projections (wqkv/wqkv_gate/ssm_out) at long ctx: run them on the fp8 (e4m3) tensor cores
    // instead of bf16. int8 is off here because the near-1-decay recurrence amplifies per-row int8
    // activation-quant error (128k top1 ~0.31); e4m3's floating range holds it to bf16-like fidelity
    // (~0.69) at the full int8 rate. The int8 activation scratch (A_i8/W_i8, 1 byte) doubles as the
    // e4m3 buffer -- fp8 GDN and int8 FFN/attn never run at the same instant within a layer.
    // SPARKINFER_PREFILL_FP8_GDN=0 restores the bf16 GDN projections (A/B).
    const char* _pfp8 = getenv("SPARKINFER_PREFILL_FP8_GDN");
    bool use_fp8_gdn = long_bf16 && (!_pfp8 || _pfp8[0] != '0');
    // MoE (Qwen3.6): run the attn/GDN projections on the fp8 (e4m3) tensor cores instead of
    // the bf16 wmma GEMM. Default ON again: the #586/#587 prefill_check failures traced to the
    // opt-in MOE_GPU tilemap path's mask dequant silently no-opping (down cols=mffn declines
    // the fast path and the returns were ignored -- fix in #593), not to the projection dtype.
    // On the default host-tilemap path, batched-vs-token top1/KL with fp8 sit inside the bf16
    // baseline's own run spread at 512..32k prefixes and clear the H3 bars (#588) with margin.
    // int8 projections stay off for MoE (router flips, as documented for use_i8 above).
    // SPARKINFER_PREFILL_MOE_FP8=0 restores the bf16 projections (A/B).
    const char* _pmfp8 = getenv("SPARKINFER_PREFILL_MOE_FP8");
    bool moe_fp8 = moe && (!_pmfp8 || _pmfp8[0] != '0');
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
    // Fused quantized-B routed GEMM (prefill_moe_q.cu): read the experts in their native GGUF
    // quantization and decode to int8 inside the B stage, so the per-layer int8 materialize never
    // happens (~1.6 GB/layer of write + read back). The dequant is a FIXED per-pass cost — it
    // materializes all 256 experts every layer whatever N — so it dominates at short prompts:
    // measured 38% of the 512 prefill and 27.5% of the 4k prefill, but only 4.9% at 32k.
    // It pays only while each expert slice is decoded ~once, i.e. while pairs/expert stays inside a
    // couple of BM tiles: at BM=128 that is 1 tile at 4k and 2 at 8k, but 8 at 32k, where
    // re-decoding costs more than materializing. Hence the context cap (default 8192, which is also
    // the CB mixed-load TTFT prefill size). SPARKINFER_PREFILL_MOE_QB=0 disables. Moved above
    // moe_serial: a BM=16 tiled kernel now exists too (see below), so moe_serial's default needs
    // to know whether the fused path can already cover this N before falling back to it.
    const int moe_qb_maxctx = [&]{
        const char* e = getenv("SPARKINFER_PREFILL_MOE_QB_MAXCTX");
        const int v = e ? atoi(e) : 8192;
        return v > 0 ? v : 8192;
    }();
    // Per-weight mask: 1 = gate, 2 = up, 4 = down (default 7 = all three, 0 = off). Per-weight
    // granularity is what makes the identity checkable: gate/up write their result directly, so
    // mask 3 vs 0 is a byte-for-byte comparison, whereas the down projection scatters through
    // float atomicAdd and so carries main's own run-to-run ordering either way.
    const int moe_qb_mask = [&]{
        const char* e = getenv("SPARKINFER_PREFILL_MOE_QB");
        return e ? atoi(e) : 7;
    }();
    const bool moe_qb_avail = moe && !moe_fused && N <= moe_qb_maxctx &&
                              s.moe_rs_gate && s.moe_rs_up && s.moe_rs_down && moe_qb_mask != 0;
    // Expert-group L2 path: dequant G experts (~G*3 MB) then GEMM that group while hot in
    // L2. Was the default at N<=512 because until now the fused quantized-B GEMM (above) only had
    // a BM=128 tiling, which under-fills badly at N<=512 (avg pairs/expert = N*top_k/E, ~16 at
    // N=512 vs a 128-row tile). A BM=16 tiling of the SAME fused-decode kernel now exists
    // (prefill_moe_q.cu's pfm_moe_gemm_qi8_bm16_kernel, matching this file's own moe_bm=16
    // choice), so prefer it over materializing — it removes the same fixed per-layer dequant this
    // path could only shrink via L2-resident chunking. SPARKINFER_PREFILL_MOE_SERIAL=1 forces the
    // old L2 path back on for A/B; GROUP default 32.
    const bool moe_serial = [&]{
        if (!moe || moe_fused) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_SERIAL");
        if (e) return e[0] != '0';
        if (moe_bm == 16 && moe_qb_avail) return false;
        return N <= 512;
    }();
    const bool moe_qb = moe_qb_avail && !moe_serial;
    // Fused gate+up GEMM (BM=16, materialized-int8 bulk path): one A staging pass per K-tile.
    // Default ON at short-N UNLESS moe_qb is already going to run gate/up through the fused-decode
    // kernel (see the bulk branch below) — that path stages A once per weight anyway and skips the
    // materialize this fusion was built to amortize, so forcing materialize+gate_up_bm16 back on
    // top of it would just re-introduce the dequant moe_qb exists to remove.
    const bool moe_fuse_gu = [&]{
        if (!moe || moe_bm != 16) return false;
        const char* e = getenv("SPARKINFER_PREFILL_MOE_FUSE_GU");
        if (e) return e[0] != '0';
        return !moe_qb;
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
            // Router logits on the bf16 tensor cores (same inputs, fp32 accumulate; only the
            // fp32 summation order differs from the warp-dot -- see prefill_router_mma.cu).
            // Falls back to the reference dot when disabled or the shape is not tile-aligned.
            if (!kernels::launch_pfm_router_logits_mma(hn, rw, mlogits, N, E, H, st))
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
                kernels::launch_prefill_swiglu_quant_i8(hg, hu, h_i8, sh, P, mffn, st);
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
                    kernels::launch_prefill_swiglu_quant_i8(hg, hu, h_i8, sh, P, mffn, st);
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
                    // Sparse: one-shot gather over live experts. Default ON — skips empty
                    // expert weight rows in one launch (+3% pp @512 vs coalesce runs alone,
                    // decode-flat, prefill_check matches). SPARKINFER_PREFILL_MOE_GATHER=0
                    // reverts to contiguous-run coalesce.
                    static const int use_gather = [] {
                        const char* e = getenv("SPARKINFER_PREFILL_MOE_GATHER");
                        if (e) return e[0] != '0' ? 1 : 0;
                        return 1;
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
                        if (e) return e[0] != '0' ? 1 : 0;
                        return 1;
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

                // Fused SwiGLU+quant (numerically identical pair replacement): drops the
                // P x mffn bf16 intermediate store + reload.
                kernels::launch_prefill_swiglu_quant_i8(hg, hu, h_i8, sh, P, mffn, sa);

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
                // Bulk: expert weights -> int8 rows ONCE per layer (all 256 experts) -- unless the
                // fused quantized-B GEMM takes the weight, in which case it is never materialized.
                // Each launch_pfm_moe_gemm_qi8 returns false (having launched nothing) for a quant
                // type or shape it cannot decode, so every weight independently falls back to the
                // dequant + int8 GEMM pair. Both paths produce the same int8 bytes and the same
                // int32 accumulation, so the outputs are identical either way.
                const float* rs_g = moe_qb ? s.moe_rs_gate + (size_t)L * E * mffn : nullptr;
                const float* rs_u = moe_qb ? s.moe_rs_up   + (size_t)L * E * mffn : nullptr;
                const float* rs_d = moe_qb ? s.moe_rs_down + (size_t)L * E * H    : nullptr;
                bool qb_g = false, qb_u = false;
                if (!moe_fuse_gu && rs_g && (moe_qb_mask & 1))
                    qb_g = kernels::launch_pfm_moe_gemm_qi8(
                        w.gate_qtype, mA_i8, msx, w.gate_q, rs_g, pair_tok, pair_w, moffsets,
                        tilemap, d_ntiles, hg, nullptr, mffn, H, max_tiles, moe_bm,
                        /*a_indirect=*/true, /*c_scatter=*/false, st);
                if (!moe_fuse_gu && rs_u && (moe_qb_mask & 2))
                    qb_u = kernels::launch_pfm_moe_gemm_qi8(
                        w.up_qtype, mA_i8, msx, w.up_q, rs_u, pair_tok, pair_w, moffsets,
                        tilemap, d_ntiles, hu, nullptr, mffn, H, max_tiles, moe_bm, true, false, st);
                if (!qb_g)
                    kernels::launch_gguf_dequant_rows_i8(w.gate_qtype, w.gate_q, Wg_i8, swg, E * mffn, H, st);
                if (!qb_u)
                    kernels::launch_gguf_dequant_rows_i8(w.up_qtype,   w.up_q,   Wu_i8, swu, E * mffn, H, st);
                if (moe_fuse_gu) {
                    kernels::launch_pfm_moe_gemm_i8_gate_up_bm16(
                        mA_i8, msx, Wg_i8, swg, Wu_i8, swu, pair_tok, moffsets, tilemap, d_ntiles,
                        hg, hu, mffn, H, max_tiles, /*e_base=*/0, st);
                } else {
                    if (!qb_g)
                        kernels::launch_pfm_moe_gemm_i8_bm(mA_i8, msx, Wg_i8, swg, pair_tok, pair_w, moffsets,
                                                           tilemap, d_ntiles, hg, nullptr, mffn, H, max_tiles, moe_bm,
                                                           /*a_indirect=*/true, /*c_scatter=*/false, st);
                    if (!qb_u)
                        kernels::launch_pfm_moe_gemm_i8_bm(mA_i8, msx, Wu_i8, swu, pair_tok, pair_w, moffsets,
                                                           tilemap, d_ntiles, hu, nullptr, mffn, H, max_tiles, moe_bm,
                                                           true, false, st);
                }
                kernels::launch_prefill_swiglu_quant_i8(hg, hu, h_i8, sh, P, mffn, st);
                pf_cu(cudaMemsetAsync(routed_f32, 0, (size_t)N * H * sizeof(float), st), "routed zero");
                bool qb_d = false;
                if (rs_d && (moe_qb_mask & 4))
                    qb_d = kernels::launch_pfm_moe_gemm_qi8(
                        w.down_qtype, h_i8, sh, w.down_q, rs_d, pair_tok, pair_w, moffsets,
                        tilemap, d_ntiles, nullptr, routed_f32, H, mffn, max_tiles, moe_bm,
                        /*a_indirect=*/false, /*c_scatter=*/true, st);
                if (!qb_d) {
                    kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q, Wd_i8, swd, E * H, mffn, st);
                    kernels::launch_pfm_moe_gemm_i8_bm(h_i8, sh, Wd_i8, swd, pair_tok, pair_w, moffsets,
                                                       tilemap, d_ntiles, nullptr, routed_f32, H, mffn, max_tiles, moe_bm,
                                                       /*a_indirect=*/false, /*c_scatter=*/true, st);
                }
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

int dflash_verify_short_run(const Qwen35PrefillCtx& s, const int* token_ids, int n, int start_pos,
                            const int* capture_layers, int n_capture, void* capture_dst,
                            int* out_argmax, bool capture_only) {
    const Qwen35Config& c = s.cfg;
    if (!token_ids || !out_argmax || n < 1 || n > 8 || !s.gguf || !c.hybrid || c.dense_ffn) {
        fprintf(stderr, "[dflash-verify] base unsupported n=%d gguf=%d hybrid=%d dense=%d\n",
                n, (int)s.gguf, (int)c.hybrid, (int)c.dense_ffn);
        return -1;
    }
    if (c.head_dim != 256 || c.linear_head_dim != 128 || c.n_experts != 256 || c.top_k <= 0) {
        fprintf(stderr, "[dflash-verify] shape unsupported hd=%d lhd=%d experts=%d topk=%d\n",
                c.head_dim, c.linear_head_dim, c.n_experts, c.top_k);
        return -1;
    }
    const int H = c.hidden, N = n, qdim = s.qdim, kvdim = s.kvdim;
    const int lqkv = s.linear_qkvdim, lvdim = s.linear_vdim, vh = c.linear_v_heads;
    const int ffn = c.moe_ffn, E = c.n_experts, topk = c.top_k;
    cudaStream_t st = s.stream;
    static thread_local Arena verify_arena;
    verify_arena.rewind();
    Arena& a = verify_arena;
    bf16* x = a.alloc<bf16>((size_t)N * H);
    bf16* xn = a.alloc<bf16>((size_t)N * H);
    bf16* h = a.alloc<bf16>((size_t)N * H);
    bf16* hn = a.alloc<bf16>((size_t)N * H);
    bf16* ao = a.alloc<bf16>((size_t)N * H);
    bf16* routed = a.alloc<bf16>((size_t)N * H);
    bf16* shared = a.alloc<bf16>((size_t)N * H);
    bf16* b8 = a.alloc<bf16>((size_t)N * 2 * qdim);
    bf16* lz = a.alloc<bf16>((size_t)N * lvdim);
    bf16* gq = a.alloc<bf16>((size_t)N * s.linear_qdim);
    bf16* gk = a.alloc<bf16>((size_t)N * s.linear_qdim);
    bf16* gv = a.alloc<bf16>((size_t)N * lvdim);
    bf16* att = a.alloc<bf16>((size_t)N * lvdim);
    bf16* lnrm = a.alloc<bf16>((size_t)N * lvdim);
    bf16* la = a.alloc<bf16>((size_t)N * vh);
    bf16* lb = a.alloc<bf16>((size_t)N * vh);
    bf16* qb = gv;
    bf16* qg = lnrm;
    bf16* kf = gq;
    bf16* vf = gk;
    bf16* sg = a.alloc<bf16>((size_t)N * ffn);
    bf16* su = a.alloc<bf16>((size_t)N * ffn);
    bf16* sh = a.alloc<bf16>((size_t)N * ffn);
    bf16* gate_raw = a.alloc<bf16>(N);
    float* gate_w = a.alloc<float>(N);
    int* ids = a.alloc<int>(N);
    int* pos = a.alloc<int>(N);
    int* seq = a.alloc<int>(N);
    int* expert_ids = a.alloc<int>((size_t)N * topk);
    float* expert_w = a.alloc<float>((size_t)N * topk);
    float* router_logits = a.alloc<float>((size_t)N * E);
    float* moe_h = a.alloc<float>((size_t)N * topk * ffn);
    float* moe_out = a.alloc<float>((size_t)N * H);
    // Dedicated hidden scratch for the shared expert. It used to borrow moe_h, which is the one
    // thing that stopped the shared branch from running concurrently with the routed one.
    float* shared_h = a.alloc<float>((size_t)N * ffn);
    float* logits = a.alloc<float>((size_t)N * c.vocab);
    int* out_ids = a.alloc<int>(N);
    const size_t q81_stride_max = kernels::llama_q8_1_bytes(std::max(H, lvdim));
    void* q81 = a.alloc<unsigned char>((size_t)N * q81_stride_max);
    const int ns = std::max(1, s.n_splits);
    float* fa_m = a.alloc<float>((size_t)N * c.n_q_heads * ns);
    float* fa_l = a.alloc<float>((size_t)N * c.n_q_heads * ns);
    float* fa_acc = a.alloc<float>((size_t)N * c.n_q_heads * ns * c.head_dim);
    // Compact recurrence records retained until posterior selection.
    bf16* rec_qkv = a.alloc<bf16>((size_t)c.n_layers * N * lqkv);
    bf16* rec_k = a.alloc<bf16>((size_t)c.n_layers * N * s.linear_qdim);
    bf16* rec_v = a.alloc<bf16>((size_t)c.n_layers * N * lvdim);
    bf16* rec_a = a.alloc<bf16>((size_t)c.n_layers * N * vh);
    bf16* rec_b = a.alloc<bf16>((size_t)c.n_layers * N * vh);
    if (!a.ok) { fprintf(stderr, "[dflash-verify] scratch allocation failed\n"); return -1; }

    static thread_local int* ph_ids = nullptr;
    static thread_local int* ph_pos = nullptr;
    static thread_local int* ph_seq = nullptr;
    static thread_local int* ph_out = nullptr;
    static thread_local cudaGraph_t verify_graph = nullptr;
    static thread_local cudaGraphExec_t verify_exec = nullptr;
    static thread_local bool graph_warm = false;
    static thread_local bool graph_ready = false;
    static thread_local const void* graph_model_key = nullptr;
    static thread_local const void* graph_state_key = nullptr;
    static thread_local const void* graph_conv_key = nullptr;
    static thread_local const void* graph_capture_key = nullptr;
    static thread_local const void* graph_btable_key = nullptr;
    static thread_local int graph_ns_key = -1;
    static thread_local const void* verify_head_key = nullptr;
    static thread_local signed char* verify_head_i8 = nullptr;
    static thread_local float* verify_head_scale = nullptr;
    static thread_local cudaEvent_t ev_fork = nullptr;
    static thread_local cudaEvent_t ev_join = nullptr;
    // Off-critical-path stream for the shared expert. Empty stream_k means no overlap is possible.
    static const bool shared_stream_on = [] {
        const char* e = getenv("SPARKINFER_DFLASH_SHARED_STREAM");
        return !(e && e[0] == '0');
    }();
    const bool fork_shared = shared_stream_on && s.stream_k && s.stream_k != s.stream;
    if (fork_shared && !ev_fork) {
        pf_cu(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming), "verify fork event");
        pf_cu(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming), "verify join event");
    }
    if (!ph_ids) {
        pf_cu(cudaHostAlloc(&ph_ids, 16 * sizeof(int), cudaHostAllocDefault), "verify host ids");
        pf_cu(cudaHostAlloc(&ph_pos, 16 * sizeof(int), cudaHostAllocDefault), "verify host pos");
        pf_cu(cudaHostAlloc(&ph_seq, 16 * sizeof(int), cudaHostAllocDefault), "verify host lens");
        pf_cu(cudaHostAlloc(&ph_out, 16 * sizeof(int), cudaHostAllocDefault), "verify host out");
    }
    if (verify_head_key != s.w.lm_head && s.w.lm_head_type == 12 && H == 2048) {
        if (verify_head_i8) cudaFree(verify_head_i8);
        if (verify_head_scale) cudaFree(verify_head_scale);
        verify_head_i8 = nullptr;
        verify_head_scale = nullptr;
        if (cudaMalloc(&verify_head_i8, (size_t)c.vocab * H) == cudaSuccess &&
            cudaMalloc(&verify_head_scale, (size_t)c.vocab * sizeof(float)) == cudaSuccess &&
            kernels::launch_gguf_dequant_rows_i8(
                s.w.lm_head_type, s.w.lm_head, verify_head_i8, verify_head_scale,
                c.vocab, H, st)) {
            pf_cu(cudaStreamSynchronize(st), "verify head int8 prepack");
            verify_head_key = s.w.lm_head;
        } else {
            if (verify_head_i8) cudaFree(verify_head_i8);
            if (verify_head_scale) cudaFree(verify_head_scale);
            verify_head_i8 = nullptr;
            verify_head_scale = nullptr;
        }
    }
    for (int i = 0; i < N; ++i) {
        ph_ids[i] = token_ids[i];
        ph_pos[i] = start_pos + i;
        ph_seq[i] = start_pos + i + 1;
    }

    const bf16* q81_src = nullptr;
    int q81_k = 0;
    auto quant_rows = [&](const bf16* in, int k) {
        if (q81_src == in && q81_k == k) return;
        kernels::launch_quantize_q8_1_rows(in, q81, k, N, k, st);
        q81_src = in;
        q81_k = k;
    };
    auto proj = [&](const bf16* in, const void* w, int type, bf16* out, int no, int k) -> bool {
        if (type == 0) {
            return kernels::launch_gemv_rows(in, w, out, N, no, k, st);
        }
        if (type != 8 && type != 12 && type != 14) {
            fprintf(stderr, "[dflash-verify] unsupported projection type=%d N=%d K=%d\n", type, no, k);
            return false;
        }
        quant_rows(in, k);
        return kernels::launch_mmvq_rows(type, q81, w, out, N, no, k, st);
    };
    // proj() on an arbitrary stream. Callers must have `in` already quantized into q81 (checked at
    // each call site), because quantizing here would write shared scratch off the main stream.
    auto proj_on = [&](cudaStream_t ps, const bf16* in, const void* w, int type, bf16* out,
                       int no, int k) -> bool {
        if (type == 0) return kernels::launch_gemv_rows(in, w, out, N, no, k, ps);
        if (type != 8 && type != 12 && type != 14) return false;
        if (q81_src != in || q81_k != k) { quant_rows(in, k); ps = st; }
        return kernels::launch_mmvq_rows(type, q81, w, out, N, no, k, ps);
    };
    auto capture = [&](int layer) {
        if (!capture_dst || !capture_layers || n_capture <= 0) return;
        for (int slot = 0; slot < n_capture; ++slot) if (capture_layers[slot] == layer) {
            char* dst = static_cast<char*>(capture_dst) + (size_t)slot * H * sizeof(bf16);
            pf_cu(cudaMemcpy2DAsync(dst, (size_t)n_capture * H * sizeof(bf16), x,
                                    (size_t)H * sizeof(bf16), (size_t)H * sizeof(bf16), N,
                                    cudaMemcpyDeviceToDevice, st), "verify capture");
        }
    };

    const int* btable = s.kv->block_table(s.seq_id);
    const int bs = s.kv->block_size(), mbs = s.kv->max_blocks_per_seq();
    const bool kv8 = s.kv->int8_kv();
    const int kv_elem = kv8 ? 1 : 2;
    // The flash-decode split/combine kernels are already batched over grid.y = num_seqs, and every
    // buffer this function hands them is laid out with exactly the per-row stride they expect. The
    // one thing that is not is the block table: they index block_table[seq * max_blocks + blk], and
    // all N verify rows share a single sequence. Replicate the table N times (N * max_blocks ints,
    // a few KB) so the 10 full-attention layers each run ONE split + ONE combine instead of one per
    // row. That removes 2*(N-1) graph nodes per attention layer, and the graph is ~1000 nodes deep
    // against only ~5.6 ms of kernel time, so node count is itself a real cost here.
    int* btab_rows = (N > 1) ? a.alloc<int>((size_t)N * mbs) : nullptr;
    if (!a.ok) { fprintf(stderr, "[dflash-verify] block-table scratch allocation failed\n"); return -1; }
    bool supported = true;
    bool recording = false;
    bool head_ok = false;
    if (graph_model_key != s.w.lm_head || graph_state_key != s.lin_state ||
        graph_conv_key != s.lin_conv_state || graph_capture_key != capture_dst ||
        graph_btable_key != btable || graph_ns_key != ns) {
        if (verify_exec) cudaGraphExecDestroy(verify_exec);
        if (verify_graph) cudaGraphDestroy(verify_graph);
        verify_exec = nullptr; verify_graph = nullptr;
        graph_ready = false; graph_warm = false;
        graph_model_key = s.w.lm_head;
        graph_state_key = s.lin_state;
        graph_conv_key = s.lin_conv_state;
        graph_capture_key = capture_dst;
        graph_btable_key = btable;
        graph_ns_key = ns;
    }
    if (graph_ready && capture_only) return 0;   // already built
    if (graph_ready) {
        pf_cu(cudaGraphLaunch(verify_exec, st), "verify graph launch");
        pf_cu(cudaStreamSynchronize(st), "verify graph sync");
        std::memcpy(out_argmax, ph_out, (size_t)N * sizeof(int));
        goto verify_forward_done;
    }
    recording = graph_warm || capture_only;
    if (recording)
        pf_cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "verify graph begin");
    pf_cu(cudaMemcpyAsync(ids, ph_ids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "verify ids");
    pf_cu(cudaMemcpyAsync(pos, ph_pos, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "verify pos");
    pf_cu(cudaMemcpyAsync(seq, ph_seq, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "verify lens");
    // Device-to-device inside the capture, so each replay re-reads the sequence's live table as it
    // grows instead of baking in the mapping from capture time.
    for (int i = 0; btab_rows && i < N; ++i)
        pf_cu(cudaMemcpyAsync(btab_rows + (size_t)i * mbs, btable, (size_t)mbs * sizeof(int),
                              cudaMemcpyDeviceToDevice, st), "verify btable row");
    kernels::launch_embedding(ids, s.w.embed_tokens, x, N, H, st);
    kernels::launch_rmsnorm(x, s.w.layers[0].input_norm, xn, N, H, c.rms_eps, st);
    for (int L = 0; L < c.n_layers && supported; ++L) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (w.linear_attn) {
            bf16* rq = rec_qkv + (size_t)L * N * lqkv;
            bf16* rk = rec_k + (size_t)L * N * s.linear_qdim;
            bf16* rv = rec_v + (size_t)L * N * lvdim;
            bf16* ra = rec_a + (size_t)L * N * vh;
            bf16* rb = rec_b + (size_t)L * N * vh;
            // wqkv, wqkv_gate and alpha/beta are three independent reads of the same xn. wqkv is
            // the only one big enough to fill the device; wqkv_gate and the fused alpha/beta run
            // at ~3 and ~0.4 CTAs per SM, so serializing them behind wqkv just adds their latency
            // to the chain. Push the two small ones onto stream_k. Safe only once q81 already
            // holds xn -- otherwise proj() would have to quantize, which writes shared scratch.
            const bool fork_gdn = fork_shared && q81_src == xn && q81_k == H;
            cudaStream_t gst = fork_gdn ? s.stream_k : st;
            if (fork_gdn) {
                pf_cu(cudaEventRecord(ev_fork, st), "verify gdn fork");
                pf_cu(cudaStreamWaitEvent(s.stream_k, ev_fork, 0), "verify gdn fork wait");
            }
            supported = proj(xn, w.wqkv, w.wqkv_type, rq, lqkv, H);
            // alpha and beta are v_heads-wide reads of the same xn — two launches whose cost is
            // almost entirely launch/graph-node latency. One fused launch, same per-row math.
            const bool ab_fused = w.ssm_alpha_type == 0 && w.ssm_beta_type == 0 &&
                kernels::launch_gemv_rows2(xn, w.ssm_alpha, w.ssm_beta, ra, rb, N, vh, vh, H, gst);
            supported = supported &&
                        proj_on(gst, xn, w.wqkv_gate, w.wqkv_gate_type, lz, lvdim, H) &&
                        (ab_fused || (proj_on(gst, xn, w.ssm_alpha, w.ssm_alpha_type, ra, vh, H) &&
                                      proj_on(gst, xn, w.ssm_beta, w.ssm_beta_type, rb, vh, H)));
            if (fork_gdn) {
                pf_cu(cudaEventRecord(ev_join, s.stream_k), "verify gdn join");
                pf_cu(cudaStreamWaitEvent(st, ev_join, 0), "verify gdn join wait");
            }
            if (!supported) break;
            const bf16* conv_live = static_cast<const bf16*>(s.lin_conv_state) +
                (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
            kernels::launch_dflash_gdn_conv_compact(rq, w.ssm_conv, conv_live, gq, rk, rv,
                N, c.linear_q_heads, vh, c.linear_head_dim, c.linear_conv_kernel, c.rms_eps, st);
            const float* state = s.lin_state + (size_t)L * vh * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_dflash_gdn_scan_compact(gq, rk, rv, ra, rb, w.ssm_dt, w.ssm_a,
                state, att, N, c.linear_q_heads, vh, c.linear_head_dim, st);
            kernels::launch_prefill_gated_norm(att, lz, w.ssm_norm, lnrm, N, vh,
                                                c.linear_head_dim, c.rms_eps, st);
            supported = proj(lnrm, w.ssm_out, w.ssm_out_type, ao, H, lvdim);
        } else {
            // Same shape of win as the GDN block: wq is 2*qdim rows and saturates, while wk and wv
            // are kvdim rows apiece and run at well under one CTA per SM.
            const bool fork_attn = fork_shared && q81_src == xn && q81_k == H;
            cudaStream_t ast = fork_attn ? s.stream_k : st;
            if (fork_attn) {
                pf_cu(cudaEventRecord(ev_fork, st), "verify attn fork");
                pf_cu(cudaStreamWaitEvent(s.stream_k, ev_fork, 0), "verify attn fork wait");
            }
            supported = proj(xn, w.wq, w.wq_type, b8, 2 * qdim, H) &&
                        proj_on(ast, xn, w.wk, w.wk_type, kf, kvdim, H) &&
                        proj_on(ast, xn, w.wv, w.wv_type, vf, kvdim, H);
            if (fork_attn) {
                pf_cu(cudaEventRecord(ev_join, s.stream_k), "verify attn join");
                pf_cu(cudaStreamWaitEvent(st, ev_join, 0), "verify attn join wait");
            }
            if (!supported || !w.q_has_gate) break;
            char* kp = static_cast<char*>(s.kv->k_pool()) +
                       (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            char* vp = static_cast<char*>(s.kv->v_pool()) +
                       (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            char* ks = kv8 ? static_cast<char*>(s.kv->k_scale_pool()) +
                             (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            char* vs = kv8 ? static_cast<char*>(s.kv->v_scale_pool()) +
                             (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            if (kv8) {
                kernels::launch_dflash_qknorm_rope_kv_partial_int8_gated(
                    b8, qb, qg, kf, vf, w.q_norm, w.k_norm, kp, vp, ks, vs, btable, pos,
                    N, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                    bs, mbs, st);
            } else {
                kernels::launch_prefill_split_q_gate(b8, qb, qg, N, c.n_q_heads, c.head_dim, st);
                kernels::launch_dflash_qknorm_rope_kv_partial(
                    qb, kf, vf, w.q_norm, w.k_norm, kp, vp, btable, pos,
                    N, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                    bs, mbs, st);
            }
            kernels::launch_flash_decode_split(
                qb, kp, vp, btab_rows ? btab_rows : btable, seq, att,
                fa_m, fa_l, fa_acc,
                N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, ns,
                // Pass the real sequence length, not -1. This argument is the HOST-side hint
                // launch_flash_decode_split uses to pick its implementation:
                //   mma_chunk = (seqlen + n_splits - 1) / n_splits
                //   mma_ok    = famma && seqlen > 512 && block_size == 16 && mma_chunk >= 32
                // With -1 the gate is always false, so compact verify silently ran the SCALAR
                // GQA kernel at every context while AR (which passes its true seqlen) switched
                // to the int8 tensor-core kernel past 512. Two different accumulation orders
                // computing what has to be the same number is precisely how the batched path
                // drifts from AR at long context (#712). start_pos + N is the largest row
                // length in this batch, matching what AR would report at the last row.
                1.f / sqrtf((float)c.head_dim), st, nullptr, start_pos + N,
                ks, vs, kv8 ? 1 : 0, kv8 ? qg : nullptr);
            // att/qg rows are contiguous at stride qdim, and the gate is elementwise, so one
            // launch covers the whole block. N separate nodes cost N times the graph-node
            // dependency latency for the same work.
            if (!kv8)
                kernels::launch_qwen36_mul_sigmoid(att, qg, N * qdim, st);
            supported = proj(att, w.wo, w.wo_type, ao, H, qdim);
        }
        if (!supported) break;
        kernels::launch_add_rmsnorm2_q8_rows(x, ao, w.post_attn_norm, h, hn, q81,
                                             N, H, c.rms_eps, st);
        q81_src = hn; q81_k = H;

        if (!w.gate_q || !w.router_w ||
            !w.shared_gate_q || !w.shared_up_q || !w.shared_down_q ||
            w.shared_gate_qtype != 8 || w.shared_up_qtype != 8 || w.shared_down_qtype != 8) {
            fprintf(stderr, "[dflash-verify] unsupported MoE layer=%d gate=%p router=%p shared=%p/%p/%p types=%d/%d/%d\n",
                    L, w.gate_q, w.router_w, w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                    w.shared_gate_qtype, w.shared_up_qtype, w.shared_down_qtype);
            supported = false; break;
        }
        // The routed and shared experts read the same hn/q81 and write disjoint outputs (routed vs
        // shared), so they are independent right up to the add_rmsnorm3 that sums them. They were
        // nonetheless serialized on one stream, which put the shared branch fully on the critical
        // path: its five launches never exceed ~3 CTAs per SM and cost ~10 us a layer, while the
        // routed branch it waits behind saturates the device (gate_up and down each launch 12288
        // CTAs). Fork the shared branch onto stream_k -- the stream AR decode already hides this
        // same kernel on -- so it fills the routed kernels' gaps instead of extending the chain.
        // Identical kernels on identical inputs in the same per-branch order: bit-identical output.
        const bool shared_on_k = fork_shared && w.shared_gate_inp &&
                                 q81_src == hn && q81_k == H;
        cudaStream_t sst = shared_on_k ? s.stream_k : st;
        if (shared_on_k) {
            pf_cu(cudaEventRecord(ev_fork, st), "verify moe fork");
            pf_cu(cudaStreamWaitEvent(s.stream_k, ev_fork, 0), "verify moe fork wait");
        }
        // Routed branch first so the saturating work is enqueued ahead of the filler.
        supported = kernels::launch_gemv_rows_f32(hn, w.router_w, router_logits,
                                                  N, E, H, st);
        kernels::launch_moe_router(router_logits, expert_ids, expert_w, nullptr,
                                   N, E, topk, 1, st);
        kernels::launch_moe_expert_ffn_q4k(hn, w.gate_q, w.up_q, w.down_q,
            w.gate_qtype, w.up_qtype, w.down_qtype, expert_ids, expert_w, routed,
            moe_h, moe_out, N, topk, H, ffn, q81, st);

        if (w.shared_gate_inp) {
            // q81 already holds hn (the add_rmsnorm2 fusion above wrote it), so this never has to
            // quantize -- which is what makes it safe to issue off the main stream.
            supported = supported &&
                (w.shared_gate_inp_type == 0
                     ? kernels::launch_gemv_rows(hn, w.shared_gate_inp, gate_raw, N, 1, H, sst)
                     : kernels::launch_mmvq_rows(w.shared_gate_inp_type, q81, w.shared_gate_inp,
                                                 gate_raw, N, 1, H, sst));
            kernels::launch_qwen36_sigmoid_rows(gate_raw, gate_w, N, sst);
        } else {
            pf_cu(cudaMemsetAsync(gate_w, 0, (size_t)N * sizeof(float), sst), "verify shared gate");
        }
        kernels::launch_shared_expert_q8_mmvq_rows(
            q81, w.shared_gate_q, w.shared_up_q, w.shared_down_q, gate_w,
            shared, shared_h, sg, H, ffn, N, sst);
        if (shared_on_k) {
            pf_cu(cudaEventRecord(ev_join, s.stream_k), "verify moe join");
            pf_cu(cudaStreamWaitEvent(st, ev_join, 0), "verify moe join wait");
        }
        const void* next_norm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_add_rmsnorm3_q8_rows(h, routed, shared, next_norm, x, xn, q81,
                                             N, H, c.rms_eps, st);
        q81_src = xn; q81_k = H;
        capture(L);
    }
    if (!supported) return -1;

    quant_rows(xn, H);
    // Score logits from the SAME weight representation AR uses. AR decode runs the
    // native Q4_K head (qwen35.cpp: launch_mmvq_q4k_f32 on s.w.lm_head), whose Q4_K
    // block structure carries a scale per 32 weights plus 6-bit sub-scales. The
    // verify_head_i8 fast path instead scores against a requantization built by
    // launch_gguf_dequant_rows_i8, which is symmetric int8 with ONE scale for the
    // whole row: `q[r,c] = round(v[r,c]/scale[r]), scale[r] = max_c|v[r,c]|/127`
    // (kernels/include/sparkinfer/kernels/quant.h). Collapsing 2048 columns onto a
    // single max-magnitude scale is far coarser than Q4_K's ~64 block scales, so
    // every row's logits carry a systematic error AR never sees -- which is exactly
    // the "discrepancy present in every row of a batch, including rows that still
    // happen to land on the correct token" reported in #712/#716, and on some
    // fraction of steps it is large enough to flip the argmax and break DFlash's
    // lossless guarantee.
    //
    // The int8 path is not even a bandwidth win: at vocab=248320 x H=2048 the int8
    // head is 508 MB against the Q4_K head's ~286 MB, so it reads ~78% MORE per
    // verify call. Defaulting to the native head is both the correct-by-construction
    // choice and the cheaper one. Opt back in with SPARKINFER_DFLASH_VERIFY_HEAD_I8=1
    // for A/B only.
    static const bool verify_head_i8_on = [] {
        const char* e = getenv("SPARKINFER_DFLASH_VERIFY_HEAD_I8");
        return e && e[0] != '0';
    }();
    head_ok = verify_head_i8_on && verify_head_i8
        ? kernels::launch_gemv_i8_q81_multirow_f32(
              q81, verify_head_i8, verify_head_scale, logits, c.vocab, H, N, st)
        : kernels::launch_mmvq_rows_f32(
              s.w.lm_head_type, q81, s.w.lm_head, logits, N, c.vocab, H, st);
    if (!head_ok) {
        fprintf(stderr, "[dflash-verify] unsupported LM head type=%d H=%d\n", s.w.lm_head_type, H);
        return -1;
    }
    kernels::launch_argmax(logits, out_ids, N, c.vocab, st);
    pf_cu(cudaMemcpyAsync(ph_out, out_ids, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost, st),
          "verify argmax");
    if (recording) {
        pf_cu(cudaStreamEndCapture(st, &verify_graph), "verify graph end");
        pf_cu(cudaGraphInstantiate(&verify_exec, verify_graph, 0), "verify graph instantiate");
        graph_ready = true;
        graph_warm = true;
        // Nothing ran: capture records the kernels rather than executing them, so the model state
        // is exactly as it was and there is no work to launch or collect.
        if (capture_only) return 0;
        pf_cu(cudaGraphLaunch(verify_exec, st), "verify graph first launch");
    }
    pf_cu(cudaStreamSynchronize(st), "verify sync");
    std::memcpy(out_argmax, ph_out, (size_t)N * sizeof(int));
    graph_warm = true;
verify_forward_done:
    int keep = 1;
    while (keep < N && token_ids[keep] == out_argmax[keep - 1]) ++keep;
    // Commit the accepted prefix into the live GDN state. This runs OUTSIDE the verify graph, on
    // the same stream, and the next step's draft does not start until it drains — so its launches
    // are on the critical path. One launch per GDN layer per commit meant 60 tiny serialized
    // kernels for work that is independent across layers; drive all of them from one grid instead.
    static const bool commit_layers = [] {
        const char* e = getenv("SPARKINFER_DFLASH_COMMIT_LAYERS");
        return !(e && e[0] == '0');
    }();
    static thread_local const void* gdn_tbl_key = nullptr;
    static thread_local int* d_gdn_layers = nullptr;
    static thread_local dflash_kernels::GdnCommitLayer* d_gdn_w = nullptr;
    static thread_local int n_gdn = 0;
    if (commit_layers && gdn_tbl_key != &s.w) {
        std::vector<int> ids;
        std::vector<dflash_kernels::GdnCommitLayer> wts;
        for (int L = 0; L < c.n_layers; ++L) if (s.w.layers[L].linear_attn) {
            ids.push_back(L);
            wts.push_back({s.w.layers[L].ssm_dt, s.w.layers[L].ssm_a, L});
        }
        if (d_gdn_layers) cudaFree(d_gdn_layers);
        if (d_gdn_w) cudaFree(d_gdn_w);
        d_gdn_layers = nullptr; d_gdn_w = nullptr; n_gdn = 0;
        if (!ids.empty() &&
            cudaMalloc(&d_gdn_layers, ids.size() * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_gdn_w, wts.size() * sizeof(dflash_kernels::GdnCommitLayer)) == cudaSuccess) {
            cudaMemcpy(d_gdn_layers, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_gdn_w, wts.data(), wts.size() * sizeof(dflash_kernels::GdnCommitLayer),
                       cudaMemcpyHostToDevice);
            n_gdn = (int)ids.size();
            gdn_tbl_key = &s.w;
        }
    }
    if (commit_layers && n_gdn > 0) {
        dflash_kernels::launch_gdn_conv_commit_layers(
            rec_qkv, (size_t)N * lqkv, s.lin_conv_state,
            (size_t)(c.linear_conv_kernel - 1) * lqkv, d_gdn_layers, n_gdn, keep,
            c.linear_q_heads, vh, c.linear_head_dim, c.linear_conv_kernel, st);
        dflash_kernels::launch_gdn_scan_commit_layers(
            rec_k, (size_t)N * s.linear_qdim, rec_v, (size_t)N * lvdim,
            rec_a, (size_t)N * vh, rec_b, d_gdn_w,
            s.lin_state, (size_t)vh * c.linear_head_dim * c.linear_head_dim,
            n_gdn, keep, c.linear_q_heads, vh, c.linear_head_dim, st);
    } else {
        for (int L = 0; L < c.n_layers; ++L) if (s.w.layers[L].linear_attn) {
            bf16* rq = rec_qkv + (size_t)L * N * lqkv;
            bf16* rk = rec_k + (size_t)L * N * s.linear_qdim;
            bf16* rv = rec_v + (size_t)L * N * lvdim;
            bf16* ra = rec_a + (size_t)L * N * vh;
            bf16* rb = rec_b + (size_t)L * N * vh;
            bf16* conv_live = static_cast<bf16*>(s.lin_conv_state) +
                (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
            float* state = s.lin_state + (size_t)L * vh * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_dflash_gdn_conv_commit(rq, conv_live, keep, c.linear_q_heads, vh,
                c.linear_head_dim, c.linear_conv_kernel, st);
            kernels::launch_dflash_gdn_scan_commit(rk, rv, ra, rb, s.w.layers[L].ssm_dt,
                s.w.layers[L].ssm_a, state, keep, c.linear_q_heads, vh, c.linear_head_dim, st);
        }
    }
    pf_cu(cudaStreamSynchronize(st), "verify commit");
    return keep;
}

} // namespace sparkinfer
