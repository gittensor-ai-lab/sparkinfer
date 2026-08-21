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
#include "sparkinfer/kernels/prefill_attn_window.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/qtype.h"
#include "sparkinfer/kernels/compressed_tensors.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/prefill_i8.h"
#include "sparkinfer/kernels/prefill_fp8.h"
#include "sparkinfer/kernels/prefill_moe.h"
#include "sparkinfer/kernels/prefill_router_mma.h"
#include "sparkinfer/kernels/prefill_moe_q.h"
#include "sparkinfer/kernels/prefill_nvfp4.h"
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
    size_t total() const { size_t t = 0; for (size_t b : sizes) t += b; return t; }
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
    // Muse Glimmer: dense hd128 GQA-16, per-layer SWA(NORMAL-rope)/global(NoPE), sandwich norm.
    // Its BF16 attention (windowed/full hd128) + bf16 KV write reuse the batched pipeline below
    // with muse-specific kernels; the full-attn scratch aliases (qb/qg<-gv/lnrm, kf/vf<-gq/gk) must
    // still fit, i.e. linear_vdim>=qdim and linear_qdim>=kvdim (4096>=4096, 2048>=256 for muse).
    // Muse runs a BF16 KV cache (its decode KV write is unconditionally bf16); if the cache is
    // int8 here the config is inconsistent -- fall back to the token loop rather than corrupt it.
    if (c.muse_glimmer) {
        if (c.head_dim != 128 || !c.dense_ffn) return -1;
        if (s.linear_vdim < s.qdim || s.linear_qdim < s.kvdim) return -1;
        if (s.kv->int8_kv()) return -1;
    } else if (c.head_dim != 256 || c.linear_head_dim != 128) {
        return -1;   // kernels specialize these
    }
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
    // Floor for the VRAM-adaptive FFN chunk below: past this the chunk loop costs more in
    // weight re-reads than the batched pass saves over the token loop.
    constexpr int kMinFfnChunk = 1024;
    // SPARKINFER_PREFILL_VERBOSE=1 reports the VRAM map around the batched scratch. Off by default.
    const bool pf_verbose = []{ const char* e = getenv("SPARKINFER_PREFILL_VERBOSE"); return e && e[0] == '1'; }();
    auto pf_vram = [&](const char* where) {
        if (!pf_verbose) return;
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        fprintf(stderr, "[prefill/vram] %-22s free=%zu MB used=%zu MB\n", where, f >> 20, (t - f) >> 20);
    };
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
    //
    // 32768 is only the STARTING value. FC is a free-VRAM-derived quantity: the sizing loop below
    // (#852, search kMinFfnChunk) halves it until the FC-scaled pair fits what cudaMemGetInfo
    // reports, so at long context it commonly lands far lower -- 1024 has been observed at
    // ctx=16384. Anything sized off FC must therefore be allocated AFTER that loop and must not
    // assume FC == N; getting this wrong is how the native-FP4 ffn_down leg silently disabled
    // itself, see the fp4_down_a comment further down.
    const int ffn_chunk = []{ const char* e = getenv("SPARKINFER_PREFILL_FFN_CHUNK"); int c = e ? atoi(e) : 32768; return c > 0 ? c : 32768; }();
    int FC = (N < ffn_chunk) ? N : ffn_chunk;
    bf16* lin_conv_state = static_cast<bf16*>(s.lin_conv_state);

    // ---- scratch ----
    // The scratch set is rebuilt from scratch on every call, and it is not small: Muse Glimmer at
    // ctx=128 asks for ~0.56 GB across ~20 buffers, and the cudaMalloc + cudaFree pair measures
    // 2.2 + 2.4 ms against a 94.5 ms prefill -- 4.6% of the batched prefill spent in the allocator,
    // on the path the harness times. Hold the arenas across calls and rewind() them instead, which
    // is what dflash_verify_short_run already does with its verify arena; the reuse check in
    // Arena::alloc keeps a buffer only while it is big enough, so a growing prefill still resizes.
    // A prefill whose scratch exceeds kArenaKeepBytes releases at the end rather than pinning it
    // (a one-off 128k prompt should not hold ~10 GB forever).
    // SPARKINFER_PREFILL_ARENA_REUSE=0 restores the per-call malloc/free (A/B).
    static const bool arena_reuse = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ARENA_REUSE");
        return !(e && e[0] == '0');
    }();
    constexpr size_t kArenaKeepBytes = 1ull << 30;
    static thread_local Arena keep_a, keep_a8, keep_am, keep_aw;   // held across calls
    Arena once_a, once_a8, once_am, once_aw;                       // per-call otherwise
    if (arena_reuse) { keep_a.rewind(); keep_a8.rewind(); keep_am.rewind(); keep_aw.rewind(); }
    Arena& a = arena_reuse ? keep_a : once_a;

    // ---- CUDA-graph replay of the whole batched prefill -------------------------------------
    // Muse's batched prefill is the only hot path here that is NOT graph-captured: decode gets a
    // graph, this issues ~1088 kernel launches per prefill@128 (20.9 per layer) with host dispatch
    // between every one. nsys puts the resulting GPU idle at 2.01 ms of a 38.18 ms window, and
    // 1076 of those 1087 gaps are under 5 us -- launch overhead, not dependency stalls.
    //
    // Capture is only legal once the arena is WARM: Arena::alloc reuses a buffer whenever it is
    // already big enough, so the second call for a given N does no cudaMalloc and hands out the
    // SAME pointers the capture recorded. Hence capture on the second sighting of an N, replay
    // after that. The prompt ids are staged through a PINNED buffer so the captured H2D has a
    // stable source address and replay picks up new ids.
    static cudaGraph_t     g_pfb_graph = nullptr;
    static cudaGraphExec_t g_pfb_exec  = nullptr;
    static int  g_pfb_n = -1, g_pfb_warm_n = -1, g_pfb_pin_cap = 0;
    static int* g_pfb_pin = nullptr;
    // DEFAULT ON: measured +5.91% on Muse prefill@128, byte-identical output (SCORE_EQ IDENTICAL,
    // TOP1 16/16). Qwen3.8-27B is the same dense-hybrid launch storm (~20 kernels/layer × 64)
    // and the same warm-arena capture is legal there — MoE stays off (host tilemaps / events).
    // SPARKINFER_MUSE_PREFILL_GRAPH=0 restores eager dispatch.
    static const bool graph_env = [] {
        const char* e = getenv("SPARKINFER_MUSE_PREFILL_GRAPH");
        return !(e && e[0] == '0');
    }();
    const bool graph_on = graph_env && arena_reuse && c.dense_ffn;
    if (graph_on && N > g_pfb_pin_cap) {
        if (g_pfb_pin) cudaFreeHost(g_pfb_pin);
        g_pfb_pin = nullptr; g_pfb_pin_cap = 0;
        if (cudaMallocHost(&g_pfb_pin, (size_t)N * sizeof(int)) == cudaSuccess) g_pfb_pin_cap = N;
    }
    const bool graph_ok = graph_on && g_pfb_pin && g_pfb_pin_cap >= N;
    if (graph_ok) memcpy(g_pfb_pin, prompt_ids, (size_t)N * sizeof(int));
    if (graph_ok && g_pfb_exec && g_pfb_n == N) {
        pf_cu(cudaGraphLaunch(g_pfb_exec, st), "pfb graph launch");
        pf_cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st),
              "pfb graph seed");
        pf_cu(cudaStreamSynchronize(st), "pfb graph sync");
        return *s.h_out_id;
    }
    pf_vram("entry");
    bf16* x    = a.alloc<bf16>((size_t)N * H);
    bf16* xn   = a.alloc<bf16>((size_t)N * H);
    bf16* hn   = a.alloc<bf16>((size_t)N * H);
    bf16* ao   = a.alloc<bf16>((size_t)N * H);
    // Muse Glimmer sandwich norm keeps the post-attention residual stream (h = x + RMSNorm(ao)*
    // post_attn_norm) live across the FFN so the post-FFN sandwich can add onto it; other models
    // fold the residual in place and need no extra buffer.
    bf16* h    = c.muse_glimmer ? a.alloc<bf16>((size_t)N * H) : nullptr;
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
    // Everything above is FC-independent and already allocated, so cudaMemGetInfo here reports
    // exactly what is left for the FC-scaled pair -- size the chunk to that instead of to a
    // constant. When ffg+ffu do not fit, the WHOLE arena alloc fails and prefill_batched returns
    // -1, so the caller silently drops to the sequential token loop: at ctx=16384 that is 79 pp
    // against 2191 pp batched, ~28x. A 27B NVFP4 checkpoint holds BOTH the NVFP4 prefill and Q4_K
    // decode weight copies, which leaves a 32 GB card short of the default chunk's 1.1 GB.
    // This only ever SHRINKS the chunk; FC < N is the same path every context above 32k already
    // takes, and the FFN is per-token independent, so it stays numerically identical.
    pf_vram("before ffg/ffu");
    // An explicit SPARKINFER_PREFILL_FFN_CHUNK is an operator decision -- honour it as given.
    if (!moe && !getenv("SPARKINFER_PREFILL_FFN_CHUNK")) {
        size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
            // What still has to come out of `fb` after the FC-scaled pair, so the chunk is sized
            // against the real remainder rather than all of free VRAM.
            const size_t tail = (size_t)maxw * sizeof(bf16)                    // wbuf
                              + (size_t)maxw                                  // W_i8
                              + (size_t)N * H                                 // A_i8 (int8) floor
                              + (size_t)N * (sizeof(float) + sizeof(int));    // sx + d_ids
            const size_t margin = (size_t)64 << 20;    // split-K partials + allocator slack
            // The chunk-parallel GDN scan draws on this same budget, AFTER this point, and it is
            // the larger consumer: its workspace is O(N) (~483 MB at ctx=16384). Sizing the FFN
            // chunk against everything that is free leaves the scan ~25 MB, which forces it into
            // ~199-token slices -- 83 per layer -- and most of the win from running it at all is
            // lost. Reserve a working segment for it here so the two are balanced rather than
            // first-come-first-served. Hybrid stacks only; nothing else runs that scan.
            //
            // This can only make the chunk SMALLER, which is the safe direction: an oversized
            // chunk is what fails the arena alloc and drops the whole pass to the token loop.
            const size_t gdn_reserve = c.hybrid ? ((size_t)256 << 20) : 0;
            const size_t claimed = tail + margin + gdn_reserve;
            const size_t avail = (fb > claimed) ? fb - claimed : 0;
            const int fc_before = FC;
            // Test the HALVED value, not the current one: `FC > floor` would step straight past it.
            while ((FC >> 1) >= kMinFfnChunk &&
                   (size_t)2 * (size_t)FC * (size_t)ffn * sizeof(bf16) > avail)
                FC >>= 1;
            if (FC != fc_before)
                fprintf(stderr, "[prefill] ffn chunk %d -> %d (ctx=%d, free=%zu MB) to keep the "
                                "batched pass\n", fc_before, FC, N, fb >> 20);
        }
    }
    bf16* ffg  = a.alloc<bf16>((size_t)FC * ffn);        // ffn gate (12288), bounded to FC tokens
    bf16* ffu  = a.alloc<bf16>((size_t)FC * ffn);        // ffn up,          bounded to FC tokens
    bf16* ffh  = ffg;                                    // SwiGLU computed in-place into ffg (down reads it)
    bf16* wbuf = a.alloc<bf16>(maxw);                    // dequantized-weight scratch (reused)
    int*  d_ids = a.alloc<int>((size_t)N);
    pf_vram("after dense arena");
    if (!a.ok) {
        // Report the numbers, not just the fact: this fallback costs ~50x at long context and the
        // old message gave no way to tell a genuinely-too-small card from a chunk set too large.
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        const size_t held = a.total();
        a.free_all();
        fprintf(stderr, "[prefill] scratch alloc failed (ctx=%d, chunk=%d, held=%zu MB, "
                        "free=%zu/%zu MB) -> fallback\n",
                N, FC, held >> 20, fb >> 20, tb >> 20);
        return -1;
    }
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
    // #845 disabled the int8 projections whenever FC < N: that combination ended the pass in an
    // illegal memory access which took the decode graph with it. The cause was the A_i8 sizing
    // below (N*H, while the o projection quantizes N*qdim), not the int8 path itself, so with the
    // sizing corrected the workaround is gone and the chunked pass keeps its tensor-core
    // projections -- worth 66.6% of the 16k pass, which was falling back to bf16 pf_gemm_kernel.
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
    Arena& a8 = arena_reuse ? keep_a8 : once_a8;
    // A_i8 holds the quantized activation. The comment below used to say the non-FFN projections
    // quantize "N rows x K(<=H)" -- they do not: the o projection's A is `att` at k = qdim, and on
    // Qwen3.8-27B qdim (24*256 = 6144) is WIDER than H (5120). N*H under-sizes it by N*(qdim-H).
    // That was invisible while FC == N, because the FC*ffn term (ffn = 17408) covered everything;
    // chunk the FFN and the cover disappears, and pf_quantize_rows_fp8_kernel writes past the end
    // (compute-sanitizer: "Invalid __global__ write ... 962561 bytes after the nearest allocation
    // of size 20971520", i.e. N*H, from the o-projection quantize at N x 6144).
    // So: N rows x the widest K any N-row projection uses, OR FC rows x ffn for the chunked FFN.
    // MoE: no chunked FFN; projections quantize N rows x maxAK.
    const bool need_i8 = use_i8 || use_i8_ffn || use_i8_attn || use_fp8_gdn || moe_shared_i8 || moe_fp8;
    // Both terms, on both dense paths: the N-row projections and the FC-row FFN chunk each have to
    // fit, and neither one bounds the other once FC and N can differ.
    const int a_wide_k = imax(H, imax(qdim, lvdim));   // widest K quantized with N rows
    const size_t a_i8_rows_n = (size_t)N * (size_t)a_wide_k;
    const size_t a_i8_dense = (a_i8_rows_n > (size_t)FC * ffn) ? a_i8_rows_n : (size_t)FC * ffn;
    const size_t a_i8_sz = moe ? (size_t)N * maxAK : a_i8_dense;
    // N, not FC: sx is the per-row scale that pairs with A_i8 above, and the non-FFN projections
    // write N of them regardless of how small the FFN chunk gets. Costs N floats.
    const size_t sx_n = (size_t)N;
    signed char* A_i8 = need_i8 ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    // k-tiled [k/32][row][32] copy of the int8 activation for Muse's dense prefill GEMM. That
    // kernel stages a 32-byte K slice of all 128 rows per pipeline step; out of the row-major A
    // those chunks are K bytes apart, so a warp issues 16 sector requests for 512 B, while out of
    // this layout the same stage is one contiguous 4 KB block -- 4 line requests. Measured on an
    // RTX 5090 by feeding the GEMM a contiguous address for the same bytes: prefill@128 30.65 ms
    // vs 32.21 ms, i.e. the scatter alone is 5% of the prefill. The quantizer writes it alongside
    // the row-major output, so nothing else has to change and every other consumer is untouched.
    // SPARKINFER_MUSE_QB_APACK=0 restores row-major staging (A/B in one binary).
    const bool muse_apack = c.muse_glimmer && need_i8 && [] {
        const char* e = getenv("SPARKINFER_MUSE_QB_APACK");
        return !(e && e[0] == '0');
    }();
    signed char* A_i8p = muse_apack ? a8.alloc<signed char>(a_i8_sz) : nullptr;
    signed char* W_i8 = need_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = need_i8 ? a8.alloc<float>(sx_n) : nullptr;
    float* sw = need_i8 ? a8.alloc<float>((size_t)maxNO) : nullptr;
    // int32 partials for the fused GEMM's split-K fan-out. Only while the whole prompt is one
    // M-tile (N <= QM_BM = 128): beyond that the plain grid already fans out over M, and the buffer
    // would scale with N. 8 * 128 * 19968 * 4 B = 82 MB, and only for Muse.
    // 13, not 8. This caps the split-K slice count, and at Muse's prefill@128 it is what the
    // ffn_down launch actually binds on: 8 slices put 8*104 = 832 blocks on a device holding 340
    // (2 blocks/SM x 170), i.e. 2.45 waves -- three waves of occupancy doing 2.45 waves of work.
    // 13 slices give 1352 blocks = 3.98 waves, so the tail wave is full. Swept 8/10/12/13/16/20;
    // 13 is the peak, and raising QM_TARGET_BLOCKS with it only adds atomic traffic (swept too).
    constexpr int QB_SPLITS = 13;
    const size_t qb_partials_cap = (size_t)QB_SPLITS * (size_t)N * (size_t)maxNO;
    // LM-head activation, quantized once (see the seed argmax at the end of this function).
    signed char* lm_q8 = a8.alloc<signed char>((size_t)H + 32);
    float* lm_ad = a8.alloc<float>((size_t)(H >> 5) + 1);
    float* lm_as = a8.alloc<float>((size_t)(H >> 5) + 1);

    int* qb_partials = (use_i8 && N <= 128 && !moe)
        ? a8.alloc<int>(qb_partials_cap) : nullptr;
    // Muse Glimmer split-K partials for the skinny projections. launch_prefill_gemm_i8 puts one
    // 128x128 output tile in a block, so Muse's narrow n_out (attn k/v = 256 -> TWO blocks, q and
    // the q-gate = 4096 -> 32) leaves the device almost empty: measured on an RTX 5090 the 2-block
    // k/v launch costs the same 69 us as the 32-block one and half of the 156-block ffn one. The
    // split-K launcher fans those over blockIdx.z and reduces int32 partials here; int32 adds are
    // exact, so the output is bit-identical (see launch_prefill_gemm_i8_splitk). Capped at
    // kSkMaxRows because the partial buffer is N*n_out int32 -- and a prefill longer than that
    // already has grid.y tiles to fill the device with, so the launcher declines it anyway.
    // SPARKINFER_PREFILL_GEMM_SPLITK=0 disables (A/B).
    constexpr int kSkMaxRows = 512;
    // Muse Glimmer and Qwen3.8-27B both score M=128, where a 128x128 tile leaves the 5090
    // empty on skinny n_out (GDN out=40 tiles, attn k/v=8). Qwen3.6 is MoE and stays off.
    const bool want_sk = !moe && need_i8 && N <= kSkMaxRows && [] {
        const char* e = getenv("SPARKINFER_PREFILL_GEMM_SPLITK");
        return !(e && e[0] == '0');
    }();
    int* sk_p = want_sk ? a8.alloc<int>((size_t)N * maxNO) : nullptr;
    if (need_i8 && !a8.ok) {
        a8.free_all();
        A_i8p = nullptr;
        use_i8 = false;
        use_i8_ffn = false;
        use_i8_attn = false;
        moe_shared_i8 = false;
        use_fp8_gdn = false;
        moe_fp8 = false;
        A_i8 = W_i8 = nullptr;
        sx = sw = nullptr;
        sk_p = nullptr;
        // free_all() above also released the LM-head seed buffers and the split-K partials, which
        // were taken from THIS arena further up -- those pointers now dangle into freed VRAM, and
        // the seed argmax at the end of this function writes through the LM-head trio
        // unconditionally. Re-take them (they are a few KB); the partials are only read on the
        // int8 path that just turned off, so drop them instead of re-allocating.
        qb_partials = nullptr;
        a8.rewind();                       // free_all() clears the buffers but leaves ok=false
        lm_q8 = a8.alloc<signed char>((size_t)H + 32);
        lm_ad = a8.alloc<float>((size_t)(H >> 5) + 1);
        lm_as = a8.alloc<float>((size_t)(H >> 5) + 1);
        if (!a8.ok) {
            a.free_all(); a8.free_all();
            fprintf(stderr, "[prefill] lm-head seed scratch alloc failed (ctx=%d) -> fallback\n", N);
            return -1;
        }
    }
    const bool mg_sk = sk_p != nullptr;

    // Native block-scaled FP4 is deliberately narrow: Muse, the scored M=128 shape, and layers
    // whose eager conversion completed. One activation buffer is shared by gate/up. Down stays on
    // the higher-fidelity #808 quantized path because its error enters the residual directly.
    const bool muse_nvfp4 = c.muse_glimmer && N == 128 && !s.w.layers.empty() &&
                            s.w.layers[0].gate_fp4 &&
                            kernels::prefill_nvfp4_supported(N, ffn, H);
    // Qwen3.8-27B's own gate/up NVFP4 (compressed-tensors checkpoint, see
    // Qwen35Model::load_compressed_tensors): same dense_ffn shape and the exact same gate_fp4/
    // up_fp4 fields Muse Glimmer already uses, so it reuses `layer_fp4`'s existing GEMM sequence
    // below unchanged -- this flag ONLY widens that one gate, mutually exclusive with
    // muse_nvfp4 at runtime (one process loads one model), so the buffers below are shared, not
    // duplicated. Deliberately NOT touching any c.muse_glimmer-gated branch elsewhere in this
    // function (qkv-fusion, wo-fusion, sandwich norm) -- those are structurally specific to
    // Muse's own tensor layout (a separate wgate tensor; Qwen3.8-27B fuses its gate into Q's
    // projection width instead) and don't apply here.
    // Muse keeps N==128 (its scored shape). Qwen3.8 now runs batched prefill at the
    // scored ctx=128 too (bf16 KV + NeoX rope, see the !kv8 arm below) as well as
    // ctx>=4096 (int8 KV). CUTLASS accepts any m%8==0, n/k%128==0.
    // DEFAULT OFF since 2026-08-16: this leg leaves the prefilled state WRONG, and does so on
    // every Qwen3.8 compressed-tensors checkpoint at every context length. Measured with
    // qwen3_gguf_prefill_check on real token ids (KV int8 off, 16 teacher-forced continuation
    // positions), unsloth/Qwen3.8-27B-NVFP4, comparing batched prefill against the token loop:
    //
    //     prompt   this leg ON            this leg OFF
    //       32     top1 12/16  KL 0.990   top1 16/16  KL 0.0024
    //      128     top1  8/16  KL 2.321   top1 16/16  KL 0.0007
    //      512     top1  6/16  KL 3.491   top1 16/16  KL 0.0026
    //
    // Off, batched prefill is exact at every length; on, the state is badly wrong and gets worse
    // with context. Turning ONLY this off also restores the ModelOpt checkpoint (KL 1.073 ->
    // 0.009), and the same model loaded from GGUF was always exact (KL 0.011) because load_gguf
    // never builds the *_fp4 operands -- which is precisely why this hid for so long.
    //
    // Nothing tested it: the PR accuracy gate runs qwen3_gguf_score.cpp, which teacher-forces
    // through forward_token() and never enters prefill_batched_run(), while the bench beside it
    // reports prefill throughput measured on exactly this path. Every prefill@128 / prefill@16k
    // number from #837 onward was measured against a wrong state.
    //
    // The defect is inside the NVFP4 GEMM sequence itself (activation quant -> gate/up GEMM ->
    // swiglu -> down GEMM), not in the weights: the same packed bytes dequantize correctly for
    // decode, and this branch is bypassed, not changed, by turning the flag off. Re-enabling it
    // needs the kernel fixed and prefill_check green at 32/128/512 -- do not flip this default
    // back on a throughput result alone, because throughput is what selected for the bug.
    // SPARKINFER_Q38_NVFP4=1 re-enables it for that debugging.
    const bool q38_nvfp4 = [&] {
        if (!c.dense_ffn || c.muse_glimmer || s.w.layers.empty() || !s.w.layers[0].gate_fp4)
            return false;
        if (!kernels::prefill_nvfp4_supported(N, ffn, H)) return false;
        const char* e = getenv("SPARKINFER_Q38_NVFP4");
        return !(e && e[0] == '0');
    }();
    const bool gu_nvfp4 = muse_nvfp4 || q38_nvfp4;
    // FP4 activation staging is sized by the FFN CHUNK, not the prompt: every consumer of these
    // buffers runs inside the token-chunked FFN loop and passes fn <= FC rows. Sizing by N asked
    // for 16x what is used at ctx=16384. Muse still gets N rows because it only reaches the FP4
    // path at N == 128, where FC == N by construction (FC = min(N, ffn_chunk)).
    // SPARKINFER_PREFILL_FP4_CHUNK_A=0 restores the N-sized buffers (A/B in ONE binary).
    static const bool fp4_chunk_a = [] {
        const char* e = getenv("SPARKINFER_PREFILL_FP4_CHUNK_A");
        return !(e && e[0] == '0');
    }();
    const int fp4_rows = fp4_chunk_a ? FC : N;
    const size_t fp4_a_data_bytes = gu_nvfp4
        ? kernels::prefill_nvfp4_data_bytes(fp4_rows, H) : 0;
    const size_t fp4_a_sf_bytes = gu_nvfp4
        ? kernels::prefill_nvfp4_scale_bytes_a(fp4_rows, H) : 0;
    // The attention projection group: q | gate | k | v stacked, so one GEMM covers all four. Its A
    // operand is `xn` at k = H -- the same shape gate/up already quantize -- so fp4_a/fp4_as serve
    // it unchanged; only the [N, qkvg_n] bf16 output and a possibly wider workspace are new.
    const int qkvg_n = 2 * qdim + 2 * kvdim;
    const bool muse_nvfp4_qkv = muse_nvfp4 && s.w.layers[0].qkvg_fp4 &&
                                kernels::prefill_nvfp4_supported(N, qkvg_n, H);
    const size_t fp4_ws_gu = gu_nvfp4
        ? kernels::prefill_nvfp4_workspace_bytes(fp4_rows, ffn, H) : 0;
    const size_t fp4_ws_qkv = muse_nvfp4_qkv
        ? kernels::prefill_nvfp4_workspace_bytes(N, qkvg_n, H) : 0;
    // o projection: A is `att` at k = qdim <= H, so fp4_a/fp4_as (sized for k = H) already cover it.
    const bool muse_nvfp4_wo = muse_nvfp4 && s.w.layers[0].wo_fp4 &&
                               kernels::prefill_nvfp4_supported(N, H, qdim);
    const size_t fp4_ws_wo = muse_nvfp4_wo
        ? kernels::prefill_nvfp4_workspace_bytes(N, H, qdim) : 0;
    const bool muse_nvfp4_down = muse_nvfp4 && s.w.layers[0].down_fp4 &&
                                 kernels::prefill_nvfp4_supported(N, H, ffn);
    const bool q38_nvfp4_down = q38_nvfp4 && s.w.layers[0].down_fp4 &&
                                kernels::prefill_nvfp4_supported(N, H, ffn);
    const bool nvfp4_down = muse_nvfp4_down || q38_nvfp4_down;
    // Same correction as fp4_down_a below: the down GEMM runs at m = fn <= FC, never at m = N.
    const size_t fp4_ws_down = nvfp4_down
        ? kernels::prefill_nvfp4_workspace_bytes(fp4_rows, H, ffn) : 0;
    size_t fp4_ws_bytes = (fp4_ws_qkv > fp4_ws_gu) ? fp4_ws_qkv : fp4_ws_gu;
    if (fp4_ws_wo > fp4_ws_bytes) fp4_ws_bytes = fp4_ws_wo;
    if (fp4_ws_down > fp4_ws_bytes) fp4_ws_bytes = fp4_ws_down;
    unsigned char* fp4_a = gu_nvfp4 ? a8.alloc<unsigned char>(fp4_a_data_bytes) : nullptr;
    unsigned char* fp4_as = gu_nvfp4 ? a8.alloc<unsigned char>(fp4_a_sf_bytes) : nullptr;
    unsigned char* fp4_ws = gu_nvfp4 ? a8.alloc<unsigned char>(fp4_ws_bytes) : nullptr;
    bf16* fp4_qkv = muse_nvfp4_qkv ? a8.alloc<bf16>((size_t)N * qkvg_n) : nullptr;
    // Sized by the FFN CHUNK, not the prompt. These two feed exactly one call --
    // launch_prefill_nvfp4_swiglu_quant_a(ffg, ffu, fp4_down_a, fp4_down_as, fn, ffn) inside the
    // token-chunked FFN loop -- so they never hold more than FC rows. Sizing them by N asked for
    // 16x what is used at ctx=16384 (142.6 MB against 8.9 MB, FC=1024 after #852's VRAM sizing),
    // the arena had nothing like that left, and a8.alloc handed back nullptr. That nullptr is not
    // an error anywhere: `down_fp4_done` just tests fp4_down_a and quietly falls through, so the
    // whole native-FP4 ffn_down leg was silently off at exactly the context it is worth the most,
    // leaving `down` on dequant-to-int8 + int8 GEMM (nsys: 6.0% + 12.9% of the prefill).
    // Muse is unaffected: it only reaches here at N == 128, where FC == N.
    unsigned char* fp4_down_a = nvfp4_down
        ? a8.alloc<unsigned char>(kernels::prefill_nvfp4_data_bytes(fp4_rows, ffn)) : nullptr;
    unsigned char* fp4_down_as = nvfp4_down
        ? a8.alloc<unsigned char>(kernels::prefill_nvfp4_scale_bytes_a(fp4_rows, ffn)) : nullptr;

    // ---- GDN projections straight off the checkpoint's NVFP4 bytes ----
    // On the ModelOpt checkpoint in_proj_qkv / in_proj_z / out_proj are NVFP4 and proj() has no
    // native-NVFP4 arm, so each one expands the weight NVFP4 -> bf16 (dq) -> int8
    // (quantize_rows_i8) -> int8 GEMM on EVERY prefill pass. That is 6.5625 bytes of traffic per
    // stored weight against the 0.5625 the payload actually occupies, over 48 GDN layers x
    // (10240 + 6144 + 6144) x 5120 weights = 5.54 G weights, i.e. ~36 GB moved per pass to read a
    // 3.1 GB operand. Feeding the packed nibbles to the same SM120 block-scaled GEMM the FFN
    // already uses deletes the expansion instead of making it faster.
    //
    // This arm was originally confined to short prompts by SPARKINFER_Q38_GDN_NVFP4_MAXN, on two
    // arguments: the saving is a per-layer FIXED cost, so it is worth the most per token exactly
    // where N is small; and the GDN recurrence amplifies activation-quant error with sequence
    // length -- this file already drops GDN off int8 past bf16_minctx for that reason, and FP4
    // activations are coarser than int8. Measurement contradicted both and the bound was lifted;
    // MAXN now defaults to effectively unlimited, and the note above gdn_fp4_maxn below records
    // what was measured. The knob remains for A/B.
    // SPARKINFER_Q38_GDN_NVFP4_PREFILL=0 restores the dequant-to-int8 path (A/B in ONE binary);
    // bit 0 is the in-projections (qkv + z, which share one A quantize) and bit 1 is out_proj, so
    // 1/2 price them separately -- they sit on opposite sides of the GDN recurrence and do not
    // carry the same activation-quant risk.
    static const int gdn_fp4_mask = [] {
        const char* e = getenv("SPARKINFER_Q38_GDN_NVFP4_PREFILL");
        return e ? atoi(e) : 3;
    }();
    const bool gdn_fp4_env = gdn_fp4_mask != 0;
    // Layer 0 is not necessarily a GDN layer (full_attention_interval), so probe for the first one.
    const Qwen35LayerWeights* gdn_probe = nullptr;
    for (const auto& lw : s.w.layers)
        if (lw.linear_attn) { gdn_probe = &lw; break; }
    // Long context takes this arm too. The 2048 bound rested on two claims, and measurement on an
    // RTX 5090 contradicts both. (a) "the saving is a per-layer FIXED cost, worth the most where N
    // is small": it is not only the avoided NVFP4->bf16->int8 expansion -- the block-scaled GEMM
    // itself replaces the int8 GEMM, and at ctx=16384 those 48x3 projections are 144 of the 208
    // pf_gemm_i8 launches that make up 22.9% of the pass (nsys), so extending the arm is worth
    // +7.7% prefill@16k (9660 -> 10404 pp, medians of 3 alternated rounds, reps=5).
    // (b) "the GDN recurrence amplifies activation-quant error with sequence length": batched
    // prefill against the token-loop reference in the same build agrees 24/24 = 1.000 at N =
    // 512 / 2048 / 8192 / 16384 with this arm on -- identical to the int8 arm at every one of
    // those lengths, so the FP4 activations do not drift as the recurrence lengthens.
    // The third claim ("a fixed bound keeps the scored ctx=128 shape off the VRAM-derived FC") is
    // untouched: 128 was already inside the bound, and prefill@128 and decode@128 both measure flat.
    static const int gdn_fp4_maxn = [] {
        const char* e = getenv("SPARKINFER_Q38_GDN_NVFP4_MAXN");
        return e ? atoi(e) : (1 << 30);
    }();
    const bool gdn_nvfp4 = gdn_fp4_env && !moe && gdn_probe && gdn_probe->gdn_qkv_fp4 &&
        N <= gdn_fp4_maxn &&
        kernels::prefill_nvfp4_supported(N, lqkv, H) &&
        kernels::prefill_nvfp4_supported(N, lvdim, H) &&
        kernels::prefill_nvfp4_supported(N, H, lvdim);
    // out_proj's A operand is `lnrm` at k = lvdim, which is WIDER than the FFN's k = H on this
    // model (6144 vs 5120), so fp4_a/fp4_as cannot be reused -- they would be overrun by a quarter
    // of a row. One staging pair sized for the widest GDN k covers all three projections.
    const int gdn_k = (lvdim > H) ? lvdim : H;
    unsigned char* fp4_gdn_a = gdn_nvfp4
        ? a8.alloc<unsigned char>(kernels::prefill_nvfp4_data_bytes(N, gdn_k)) : nullptr;
    unsigned char* fp4_gdn_as = gdn_nvfp4
        ? a8.alloc<unsigned char>(kernels::prefill_nvfp4_scale_bytes_a(N, gdn_k)) : nullptr;
    // The three GDN shapes can each want more workspace than the FFN's, and fp4_ws is shared.
    unsigned char* fp4_gdn_ws = nullptr;
    if (gdn_nvfp4) {
        size_t wb = kernels::prefill_nvfp4_workspace_bytes(N, lqkv, H);
        const size_t wz = kernels::prefill_nvfp4_workspace_bytes(N, lvdim, H);
        const size_t wo = kernels::prefill_nvfp4_workspace_bytes(N, H, lvdim);
        if (wz > wb) wb = wz;
        if (wo > wb) wb = wo;
        fp4_gdn_ws = (wb <= fp4_ws_bytes && fp4_ws) ? fp4_ws : a8.alloc<unsigned char>(wb);
    }

    // ---- full-attention q|gate / k / v / o straight off the checkpoint's NVFP4 bytes ----
    // The 16 softmax-attention layers were the last projections still leaving the block-scaled
    // GEMM: proj() ran them as a dp4a int8 GEMM over the Q4_K copy, measured at ~30% of peak
    // against the FP4 GEMM's 76% beside it (nsys, ctx=128: 1.076 ms for q|gate,k,v + 0.589 ms for
    // o, per pass, moving 0.94 GB). Loading them with keep_native (qwen35.cpp) makes the packed
    // nibbles the resident form, so this is the same GEMM the FFN and GDN already use.
    //
    // Bounded by N like the GDN arm and for the same reason -- the saving is per-layer fixed cost,
    // so it is worth the most where N is small -- and so the scored ctx=128 shape cannot depend on
    // the cudaMemGetInfo-derived FC. Bit 0 is q|gate/k/v (which share one A quantize), bit 1 is o.
    static const int attn_fp4_mask = [] {
        const char* e = getenv("SPARKINFER_Q38_ATTN_NVFP4_PREFILL");
        return e ? atoi(e) : 3;
    }();
    // Unlike the GDN arm this is NOT bounded to short prompts. #860 bounds GDN at 2048 because
    // the delta-rule recurrence compounds activation-quant error along the sequence; softmax
    // attention has no such carry -- each q/k/v row is projected independently -- so the FP4
    // activation error does not accumulate with N. It also MUST cover every length here: the Q4_K
    // copy no longer exists, so any N that misses this arm falls back to expanding the NVFP4
    // weight to bf16 and then to int8 on every pass, which is far worse than the arm is good
    // (measured: -3.7% at ctx=16384 when the bound was left at 2048).
    static const int attn_fp4_maxn = [] {
        const char* e = getenv("SPARKINFER_Q38_ATTN_NVFP4_MAXN");
        return e ? atoi(e) : (1 << 30);
    }();
    // Layer 0 is a GDN layer under full_attention_interval, so probe for the first full-attn one.
    const Qwen35LayerWeights* attn_probe = nullptr;
    for (const auto& lw : s.w.layers)
        if (!lw.linear_attn) { attn_probe = &lw; break; }
    const int wide_n = 2 * qdim;                     // wq holds [q|gate] as one operand
    const bool attn_nvfp4 = attn_fp4_mask != 0 && !moe && !c.muse_glimmer &&
        attn_probe && attn_probe->wq_fp4 && N <= attn_fp4_maxn &&
        kernels::prefill_nvfp4_supported(N, wide_n, H) &&
        kernels::prefill_nvfp4_supported(N, kvdim, H) &&
        kernels::prefill_nvfp4_supported(N, H, qdim);
    // o's A operand is `att` at k = qdim (6144 here), wider than the q/k/v k = H, so one staging
    // pair sized for the widest of the two covers all four projections.
    const int attn_k = (qdim > H) ? qdim : H;
    unsigned char* fp4_attn_a = attn_nvfp4
        ? a8.alloc<unsigned char>(kernels::prefill_nvfp4_data_bytes(N, attn_k)) : nullptr;
    unsigned char* fp4_attn_as = attn_nvfp4
        ? a8.alloc<unsigned char>(kernels::prefill_nvfp4_scale_bytes_a(N, attn_k)) : nullptr;
    unsigned char* fp4_attn_ws = nullptr;
    if (attn_nvfp4) {
        size_t wb = kernels::prefill_nvfp4_workspace_bytes(N, wide_n, H);
        const size_t wk = kernels::prefill_nvfp4_workspace_bytes(N, kvdim, H);
        const size_t wo2 = kernels::prefill_nvfp4_workspace_bytes(N, H, qdim);
        if (wk > wb) wb = wk;
        if (wo2 > wb) wb = wo2;
        fp4_attn_ws = (wb <= fp4_ws_bytes && fp4_ws) ? fp4_ws : a8.alloc<unsigned char>(wb);
    }

    // Long-ctx FFN int8: keep gate/up/down int8 weights (+scales) across token chunks so each
    // layer dequants once instead of once per chunk. ~150 MB vs ~300 MB for a bf16 cache.
    Arena& aw = arena_reuse ? keep_aw : once_aw;
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
    Arena& am = arena_reuse ? keep_am : once_am;
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

    // Capture on the SECOND sighting of this N (arena warm => no cudaMalloc inside the capture).
    bool pfb_capturing = false;
    // Re-capture when N changes: a graph is only valid for the N it recorded.
    if (graph_ok && g_pfb_exec && g_pfb_n != N) {
        cudaGraphExecDestroy(g_pfb_exec); g_pfb_exec = nullptr;
        if (g_pfb_graph) { cudaGraphDestroy(g_pfb_graph); g_pfb_graph = nullptr; }
        g_pfb_n = -1;
    }
    if (graph_ok && !g_pfb_exec && g_pfb_warm_n == N) {
        if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) == cudaSuccess)
            pfb_capturing = true;
    }
    pf_cu(cudaMemcpyAsync(d_ids, graph_ok ? g_pfb_pin : prompt_ids, (size_t)N * sizeof(int),
                          cudaMemcpyHostToDevice, st), "prefill ids");

    // Dequantize a native GGUF weight [n_out,K] to bf16 scratch; return a bf16 [n_out,K] ptr.
    auto dq = [&](const void* W, int wtype, int n_out, int K) -> const void* {
        if (wtype == 0) return W;   // already bf16 dense
        if (wtype == kernels::SI_QTYPE_FP8) {
            kernels::launch_ct_dequant_fp8_packed(W, wbuf, n_out, K, st);
            return wbuf;
        }
        if (wtype == kernels::SI_QTYPE_NVFP4) {
            // Payload is [256 B header with the f32 global scale | ue4m3 group scales | packed
            // nibbles]; the global scale lives on the device, so read it through the kernel's own
            // pointer rather than copying it back to the host inside a prefill step.
            const size_t hdr = (size_t)kernels::SI_NVFP4_HDR;
            const size_t scale_bytes = (size_t)n_out * K / 16;
            kernels::launch_ct_dequant_nvfp4_dev(
                static_cast<const char*>(W) + hdr + scale_bytes,
                static_cast<const char*>(W) + hdr,
                static_cast<const float*>(W), wbuf, n_out, K, st);
            return wbuf;
        }
        kernels::launch_gguf_dequant(wtype, W, wbuf, (long)n_out * K, st);
        return wbuf;
    };
    // int8 activation memo: consecutive int8 projections of the SAME input (wq/wk/wv on xn,
    // wqkv/wqkv_gate on xn, FFN gate/up on the same chunk) re-quantize identical values into
    // A_i8/sx each call. Remember what A_i8 currently holds and skip the repeat quantize --
    // bit-identical, it reuses the exact bytes the first call produced. The memo is reset at
    // every layer top (xn/hn refresh in place) and wherever A_i8 is written outside proj().
    const bf16* a_q = nullptr; int a_qR = 0, a_qK = 0;
    // Whether A_i8p currently holds the k-tiled copy of what is in A_i8. Every writer of A_i8
    // either refreshes it or clears this, so a stale copy can never reach the GEMM.
    bool a_pk = false;
    auto apk = [&]() -> const signed char* { return a_pk ? A_i8p : nullptr; };
    auto quant_a_i8 = [&](const bf16* A, int R, int K) {
        if (a_q == A && a_qR == R && a_qK == K) return;
        a_pk = kernels::launch_prefill_quantize_rows_i8(A, A_i8, sx, R, K, st, A_i8p) && A_i8p;
        a_q = A; a_qR = R; a_qK = K;
    };
    // int8 tensor-core GEMM with the Muse-only split-K fan-out tried first. Everything else keeps
    // calling the single-block launcher, so non-Muse output is byte-for-byte what it was.
    auto gemm_i8 = [&](const signed char* Aq, const signed char* Wq, const float* sxq,
                       const float* swq, bf16* Cc, int R, int n_out, int K, bool resid) {
        if (mg_sk && kernels::launch_prefill_gemm_i8_splitk(Aq, Wq, sxq, swq, Cc, R, n_out, K,
                                                            sk_p, resid, st))
            return;
        if (resid) kernels::launch_prefill_gemm_i8_resid(Aq, Wq, sxq, swq, Cc, R, n_out, K, st);
        else       kernels::launch_prefill_gemm_i8(Aq, Wq, sxq, swq, Cc, R, n_out, K, st);
    };
    // C[N,n_out] = A[N,K] @ W^T  (W native quantized [n_out,K]).
    static const bool q38_fp8_prefill = [] {
        const char* e = getenv("SPARKINFER_Q38_FP8_PREFILL");
        return !(e && e[0] == '0');
    }();
    auto proj_fp8_native = [&](const bf16* A, const void* W, bf16* C, int R, int n_out, int K) -> bool {
        // Checkpoint SI_QTYPE_FP8 is already e4m3 + per-row bf16 scale. At the scored
        // ctx=128 the int8 arm dequants that to bf16 and requants to s8 every GDN
        // projection (48 layers × qkv/z/out). Feed the packed e4m3 to the existing
        // fp8 GEMM instead. SPARKINFER_Q38_FP8_PREFILL=0 restores the requant path.
        if (!q38_fp8_prefill || !W || !A_i8 || !sx || !sw || n_out < 128) return false;
        a_q = nullptr; a_pk = false;
        kernels::launch_prefill_quantize_rows_fp8(A, A_i8, sx, R, K, st);
        kernels::launch_prefill_fp8_wscales_bf16(W, sw, n_out, st);
        const void* We4 = static_cast<const char*>(W) + (size_t)n_out * 2;
        if (!(sk_p && kernels::launch_prefill_gemm_fp8_splitk(
                A_i8, We4, sx, sw, C, R, n_out, K,
                reinterpret_cast<float*>(sk_p), st)))
            kernels::launch_prefill_gemm_fp8(A_i8, We4, sx, sw, C, R, n_out, K, st);
        return true;
    };
    auto proj = [&](const bf16* A, const void* W, int wtype, bf16* C, int n_out, int K, int rows = 0) {
        const int R = rows > 0 ? rows : N;   // rows (M) to process; chunked FFN passes a sub-N count
        if (wtype == kernels::SI_QTYPE_FP8 && proj_fp8_native(A, W, C, R, n_out, K)) return;
        // int8 only for the big weight-bound projections; keep the tiny per-v-head gate
        // projections (ssm_alpha/ssm_beta, n_out == v_heads) in bf16 — they feed the GDN
        // sigmoid gates, where per-row int8 quant of a 32-wide weight costs more accuracy
        // than the negligible time it saves.
        if (use_i8 && n_out >= 128) {
            quant_a_i8(A, R, K);
            // fused Q4_K/Q6_K -> int8 rows skips the dequant-to-bf16 scratch round trip
            bool w_i8_ready = kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st);
            // Same for a checkpoint NVFP4 weight, which otherwise has no fused arm at all: dq()
            // writes the whole [n_out,K] as bf16 and the row-quantizer reads it straight back.
            // Bit-identical to that pair -- see launch_ct_dequant_nvfp4_rows_i8.
            if (!w_i8_ready && wtype == kernels::SI_QTYPE_NVFP4) {
                const size_t hdr = (size_t)kernels::SI_NVFP4_HDR;
                const size_t scale_bytes = (size_t)n_out * K / 16;
                w_i8_ready = kernels::launch_ct_dequant_nvfp4_rows_i8(
                    static_cast<const char*>(W) + hdr + scale_bytes,
                    static_cast<const char*>(W) + hdr,
                    static_cast<const float*>(W), W_i8, sw, n_out, K, st);
            }
            if (!w_i8_ready) {
                const void* wb = dq(W, wtype, n_out, K);
                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
            }
            gemm_i8(A_i8, W_i8, sx, sw, C, R, n_out, K, false);
        } else if ((use_fp8_gdn || moe_fp8) && n_out >= 128) {
            // fp8 (e4m3) tensor-core path for the long-ctx GDN projections. A_i8/W_i8 (1 byte) hold
            // the e4m3 operands; dequant the weight to bf16 scratch, then row/channel fp8-quantize.
            a_q = nullptr; a_pk = false;                // A_i8 becomes e4m3 -- invalidate the memo
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
    // Same idea for the block-scaled NVFP4 projections, which used to be excluded from it because
    // the launcher had no source operand: CUTLASS's LinearCombination epilogue is already
    // D = alpha*Acc + beta*C, so passing the residual as C and beta=1 folds the add in.
    // SPARKINFER_PREFILL_NVFP4_RESID_FUSE=0 restores the raw-projection + separate-add form (A/B).
    const bool nvfp4_resid_fuse = resid_fuse && [] {
        const char* e = getenv("SPARKINFER_PREFILL_NVFP4_RESID_FUSE");
        return !(e && e[0] == '0');
    }();
    auto proj_resid = [&](const bf16* A, const void* W, int wtype, bf16* Cx, int n_out, int K,
                          int rows = 0) -> bool {
        if (!resid_fuse || !use_i8 || n_out < 128 || wtype == kernels::SI_QTYPE_FP8) return false;
        const int R = rows > 0 ? rows : N;
        quant_a_i8(A, R, K);
        if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
            const void* wb = dq(W, wtype, n_out, K);
            kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
        }
        gemm_i8(A_i8, W_i8, sx, sw, Cx, R, n_out, K, true);
        return true;
    };

    // Muse Glimmer fused quantized-B projection: decode the native Q4_K/Q5_K weight to int8 INSIDE
    // the GEMM (launch_prefill_gemm_qi8_dense) using the per-row scale precomputed at load (w.*_rs),
    // skipping the int8 materialize (dequant -> W_i8 -> reload) that proj()'s int8 path pays -- the
    // dominant weight-bandwidth cost at prefill's M=128, where the GEMM tile is compute-bound and the
    // materialize round-trip is pure overhead. Bit-identical to proj(): same activation int8 (shared
    // quant_a_i8 memo), same weight decode + per-row scale + int8 bytes, same int8 tensor-core
    // accumulation. Falls back to proj() when the scale is absent (Q6_K down, precompute off) or the
    // weight type/shape is unsupported. Reaches ONLY Muse (c.muse_glimmer); every other model's proj()
    // calls are untouched. SPARKINFER_MUSE_PREFILL_QB=0 forces the materialize path (A/B).
    const bool muse_qb = c.muse_glimmer && [] {
        const char* e = getenv("SPARKINFER_MUSE_PREFILL_QB");
        return !(e && e[0] == '0');
    }();
    const bool muse_group = c.muse_glimmer &&
        [] { const char* e = getenv("SPARKINFER_MUSE_PREFILL_GROUP"); return !(e && e[0] == '0'); }();
    // SPARKINFER_MUSE_FFN_GROUP=0 keeps ffn gate/up as two launches (A/B).
    const bool muse_ffn_group = c.muse_glimmer &&
        [] { const char* e = getenv("SPARKINFER_MUSE_FFN_GROUP"); return !(e && e[0] == '0'); }();
    // SPARKINFER_MUSE_GROUP_SUBSET=0 goes back to grouping only when all four projections share a
    // type, which is the all-or-nothing test this replaces (A/B).
    const bool muse_gsubset =
        [] { const char* e = getenv("SPARKINFER_MUSE_GROUP_SUBSET"); return !(e && e[0] == '0'); }();
    // Same as proj_fused, but lets the split-K accumulator stand instead of reducing it to bf16,
    // for the two projections (o and ffn_down) whose only consumer is the sandwich norm. Sets
    // *acc to 1 when it did; the caller then feeds qb_partials to launch_norm_then_add_acc.
    auto proj_fused_acc = [&](const bf16* A, const void* W, int wtype, const float* rs,
                              bf16* C, int n_out, int K, int* acc, int rows = 0) {
        const int R = rows > 0 ? rows : N;
        if (use_i8 && rs && n_out >= 128 && kernels::pf_dense_gemm_qi8_supported(wtype)) {
            quant_a_i8(A, R, K);
            if (kernels::launch_prefill_gemm_qi8_dense(wtype, A_i8, sx, W, rs, C, R, n_out, K, st,
                                                       qb_partials, QB_SPLITS, acc, apk()))
                return;
        }
        proj(A, W, wtype, C, n_out, K, rows);
    };
    auto proj_fused = [&](const bf16* A, const void* W, int wtype, const float* rs,
                          bf16* C, int n_out, int K, int rows = 0) {
        const int R = rows > 0 ? rows : N;
        if (use_i8 && rs && n_out >= 128 && kernels::pf_dense_gemm_qi8_supported(wtype)) {
            quant_a_i8(A, R, K);
            if (kernels::launch_prefill_gemm_qi8_dense(wtype, A_i8, sx, W, rs, C, R, n_out, K, st,
                                                       qb_partials, QB_SPLITS, nullptr, apk()))
                return;
        }
        proj(A, W, wtype, C, n_out, K, rows);
    };

    // GDN wqkv + wqkv_gate both project the same input xn, so on the fp8 path quantize xn to e4m3
    // ONCE and share it across both GEMMs (proj() would otherwise re-quantize xn per projection --
    // a full redundant read of xn and rewrite of the e4m3 activation each layer). Bit-identical to
    // the two independent proj() calls. Default on with either fp8 projection path (dense >96k GDN
    // or MoE); SPARKINFER_PREFILL_FP8_GDN_SHAREQ=0 restores the per-projection quantize (A/B).
    const char* _pshareq = getenv("SPARKINFER_PREFILL_FP8_GDN_SHAREQ");
    const bool fp8_shareq = (use_fp8_gdn || moe_fp8) && (!_pshareq || _pshareq[0] != '0');
    auto gdn_qkv_z = [&](const bf16* A, const Qwen35LayerWeights& w) {
        // Checkpoint-native NVFP4: quantize xn to FP4 ONCE (both projections read it) and run two
        // block-scaled GEMMs straight off the packed nibbles. A_i8/sx are not touched, so the int8
        // activation memo stays valid for whatever runs next in the layer.
        if (gdn_nvfp4 && (gdn_fp4_mask & 1) &&
            w.gdn_qkv_fp4 && w.gdn_qkv_fp4_sf && w.gdn_z_fp4 && w.gdn_z_fp4_sf &&
            fp4_gdn_a && fp4_gdn_as && fp4_gdn_ws &&
            kernels::launch_prefill_nvfp4_quant_a(A, fp4_gdn_a, fp4_gdn_as, N, H, st) &&
            kernels::launch_prefill_nvfp4_gemm(fp4_gdn_a, fp4_gdn_as,
                                               w.gdn_qkv_fp4, w.gdn_qkv_fp4_sf,
                                               b8, N, lqkv, H, fp4_gdn_ws, st,
                                               w.gdn_qkv_fp4_alpha) &&
            kernels::launch_prefill_nvfp4_gemm(fp4_gdn_a, fp4_gdn_as,
                                               w.gdn_z_fp4, w.gdn_z_fp4_sf,
                                               lz, N, lvdim, H, fp4_gdn_ws, st,
                                               w.gdn_z_fp4_alpha))
            return;
        if (q38_fp8_prefill && w.wqkv_type == kernels::SI_QTYPE_FP8 &&
            w.wqkv_gate_type == kernels::SI_QTYPE_FP8 && A_i8 && sx && sw) {
            a_q = nullptr; a_pk = false;
            kernels::launch_prefill_quantize_rows_fp8(A, A_i8, sx, N, H, st);
            kernels::launch_prefill_fp8_wscales_bf16(w.wqkv, sw, lqkv, st);
            const void* Wq = static_cast<const char*>(w.wqkv) + (size_t)lqkv * 2;
            if (!(sk_p && kernels::launch_prefill_gemm_fp8_splitk(
                    A_i8, Wq, sx, sw, b8, N, lqkv, H,
                    reinterpret_cast<float*>(sk_p), st)))
                kernels::launch_prefill_gemm_fp8(A_i8, Wq, sx, sw, b8, N, lqkv, H, st);
            kernels::launch_prefill_fp8_wscales_bf16(w.wqkv_gate, sw, lvdim, st);
            const void* Wz = static_cast<const char*>(w.wqkv_gate) + (size_t)lvdim * 2;
            if (!(sk_p && kernels::launch_prefill_gemm_fp8_splitk(
                    A_i8, Wz, sx, sw, lz, N, lvdim, H,
                    reinterpret_cast<float*>(sk_p), st)))
                kernels::launch_prefill_gemm_fp8(A_i8, Wz, sx, sw, lz, N, lvdim, H, st);
        } else if (fp8_shareq) {
            a_q = nullptr; a_pk = false;                // A_i8 becomes e4m3 -- invalidate the memo
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
    // Muse Glimmer: unweighted embedding RMSNorm on x before layer 0 (decode qwen35.cpp:724-725).
    // emb_norm_ones is a constant-1.0 weight, so this is a pure normalization of the embedding.
    if (c.muse_glimmer && s.emb_norm_ones)
        kernels::launch_rmsnorm(x, (const bf16*)s.emb_norm_ones, x, N, H, eps, st);
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
        a_q = nullptr; a_pk = false;                   // xn/hn are refreshed in place each layer
        bool attn_fused = false;                       // post-attn residual folded into the proj?
        // Set when the o / ffn_down split-K accumulator was left un-reduced for the sandwich
        // norm to consume directly. Per layer: qb_partials is reused by the next GEMM.
        int attn_acc = 0, ffn_acc = 0;
        bool hn_quantized = false;   // pre-FFN norm already emitted A_i8/sx for the grouped FFN
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
                layer_state, att, N, c.linear_q_heads, vh, c.linear_head_dim,
                c.gdn_qh_block, st);
            kernels::launch_prefill_gated_norm(att, lz, w.ssm_norm, lnrm, N, vh, c.linear_head_dim, eps, st);
            // out_proj off the same NVFP4 bytes, with the residual folded into the block-scaled
            // GEMM's own epilogue (D = A*B + C, C aliasing D aliasing x) instead of written raw
            // to `ao` for a separate full-tensor add. That add is three N*H bf16 streams (read x,
            // read ao, write x) for one flop per element; the epilogue already holds the
            // accumulator in registers, so folding it in costs one read and removes the pass.
            // The fused form claims the residual itself -- attn_fused suppresses the add below.
            bool out_fp4 = false;
            if (gdn_nvfp4 && (gdn_fp4_mask & 2) && w.gdn_out_fp4 && w.gdn_out_fp4_sf &&
                fp4_gdn_a && fp4_gdn_as && fp4_gdn_ws &&
                kernels::launch_prefill_nvfp4_quant_a(lnrm, fp4_gdn_a, fp4_gdn_as, N, lvdim, st)) {
                if (nvfp4_resid_fuse &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_gdn_a, fp4_gdn_as,
                                                       w.gdn_out_fp4, w.gdn_out_fp4_sf,
                                                       x, N, H, lvdim, fp4_gdn_ws, st,
                                                       w.gdn_out_fp4_alpha, x)) {
                    out_fp4 = true;
                    attn_fused = true;
                } else if (kernels::launch_prefill_nvfp4_gemm(fp4_gdn_a, fp4_gdn_as,
                                                              w.gdn_out_fp4, w.gdn_out_fp4_sf,
                                                              ao, N, H, lvdim, fp4_gdn_ws, st,
                                                              w.gdn_out_fp4_alpha)) {
                    out_fp4 = true;
                }
            }
            if (!out_fp4) {
                attn_fused = proj_resid(lnrm, w.ssm_out, w.ssm_out_type, x, H, lvdim);
                if (!attn_fused) proj(lnrm, w.ssm_out, w.ssm_out_type, ao, H, lvdim);
            }
            use_i8 = restore_i8_gdn;
        } else {
            // ---- full softmax-attention layer (q_has_gate, partial RoPE, int8 KV) ----
            // Long-ctx: optionally keep Q/K/V/O on int8 (no GDN recurrence here).
            const bool restore_i8 = use_i8;
            if (use_i8_attn) use_i8 = true;
            if (c.muse_glimmer && w.wgate) {
                // Muse Glimmer keeps attn_gate as its OWN quantized tensor (w.wgate, [qdim,H]) -- Q
                // goes straight to qb and the gate straight to qg, with no [q|gate] interleave to build
                // and no split to undo (matches decode's sep_gate path, qwen35.cpp:988-1007). Projecting
                // w.wq as `wide` (2*qdim) would dequant-read qdim rows PAST the 4096-row wq tensor.
                // q/gate/k/v all read xn and are mutually independent. At M=128 each of them is
                // ceil(n_out/64) CTAs of a 170-SM machine -- 64, 64, 4, 4 -- so as four launches
                // they cost four full CTA-durations, two of which carry four CTAs of work. One
                // grid for all four collapses that to one. Bit-identical per output tile.
                // SPARKINFER_MUSE_PREFILL_GROUP=0 restores the four separate launches.
                // Group whichever of the four share the anchor type instead of demanding that all
                // four match: this GGUF gives half its layers a Q6_K attn_v, and the all-or-nothing
                // test dropped every one of those layers back to four separate launches -- q and
                // gate at 64 tiles, k at 4. Anything left out (or unsupported) still goes through
                // proj_fused, the same path it took before, so its result is unchanged.
                const void* Wa[4]; const float* rsa[4]; void* Ca[4]; int na[4];
                bool inq = false, ing = false, ink = false, inv = false;
                int ng = 0;
                // FP4 form of that same grouped GEMM: q|gate|k|v are one [qkvg_n, H] operand, so
                // this is a single block-scaled GEMM into a contiguous [N, qkvg_n] buffer, then
                // four strided copies out to the destinations the rest of the layer expects
                // (they are aliases into the GDN scratch and each wants a tight row stride).
                // 2.2 MB of copy per layer against ~25 MB less weight traffic.
                bool qkvg_fp4 = false;
                if (muse_nvfp4_qkv && w.qkvg_fp4 && w.qkvg_fp4_sf && fp4_a && fp4_as && fp4_qkv &&
                    kernels::launch_prefill_nvfp4_quant_a(xn, fp4_a, fp4_as, N, H, st) &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_a, fp4_as, w.qkvg_fp4, w.qkvg_fp4_sf,
                                                       fp4_qkv, N, qkvg_n, H, fp4_ws, st)) {
                    const size_t sp = (size_t)qkvg_n * sizeof(bf16);
                    struct { void* dst; int off, n; } cut[4] = {
                        { qb, 0,            qdim  }, { qg, qdim,          qdim  },
                        { kf, 2 * qdim,     kvdim }, { vf, 2 * qdim + kvdim, kvdim },
                    };
                    qkvg_fp4 = true;
                    for (auto& cu : cut)
                        qkvg_fp4 &= cudaMemcpy2DAsync(cu.dst, (size_t)cu.n * sizeof(bf16),
                                                      fp4_qkv + cu.off, sp,
                                                      (size_t)cu.n * sizeof(bf16), N,
                                                      cudaMemcpyDeviceToDevice, st) == cudaSuccess;
                    if (qkvg_fp4) inq = ing = ink = inv = true;
                }
                if (!qkvg_fp4 && muse_group && muse_qb && use_i8 &&
                    kernels::pfm_moe_gemm_qi8_supported(w.wq_type)) {
                    const int at = w.wq_type;
                    auto take = [&](const void* W, int t, const float* rs, void* C, int n, bool& in) {
                        if (!W || !rs || t != at) return;
                        Wa[ng] = W; rsa[ng] = rs; Ca[ng] = C; na[ng] = n; ng++; in = true;
                    };
                    take(w.wq,    w.wq_type,    w.wq_rs,    qb, qdim,  inq);
                    take(w.wgate, w.wgate_type, w.wgate_rs, qg, qdim,  ing);
                    take(w.wk,    w.wk_type,    w.wk_rs,    kf, kvdim, ink);
                    take(w.wv,    w.wv_type,    w.wv_rs,    vf, kvdim, inv);
                    if (!muse_gsubset && ng != 4) { ng = 0; inq = ing = ink = inv = false; }
                }
                bool grouped = false;
                if (ng >= 2) {
                    quant_a_i8(xn, N, H);
                    grouped = kernels::launch_prefill_gemm_qi8_dense_group(
                        w.wq_type, A_i8, sx, Wa, rsa, Ca, na, ng, N, H, st,
                        qb_partials, QB_SPLITS, qb_partials_cap,
                        nullptr, nullptr, nullptr, apk());
                }
                if (!grouped && !qkvg_fp4) { inq = ing = ink = inv = false; }
                if (!inq) proj_fused(xn, w.wq,    w.wq_type,    w.wq_rs,    qb, qdim,  H);
                if (!ing) proj_fused(xn, w.wgate, w.wgate_type, w.wgate_rs, qg, qdim,  H);
                if (!ink) proj_fused(xn, w.wk,    w.wk_type,    w.wk_rs,    kf, kvdim, H);
                if (!inv) proj_fused(xn, w.wv,    w.wv_type,    w.wv_rs,    vf, kvdim, H);
            } else {
                // Qwen3.8: [q|gate] is one wide wq, then skinny k/v (8 tiles each). One grouped
                // launch fills the 5090; four separate ones leave k/v paying a full CTA duration
                // for 8 blocks. Same kernel as Muse, bit-identical per tile.
                // Checkpoint-native NVFP4: quantize xn to FP4 ONCE (all three projections read it)
                // and run three block-scaled GEMMs off the packed nibbles. [q|gate] stays the one
                // wide operand it already is, so the split below is untouched. A_i8/sx are not
                // written, so the int8 activation memo stays valid for whatever runs next.
                bool qkv_fp4 = false;
                if (attn_nvfp4 && (attn_fp4_mask & 1) &&
                    w.wq_fp4 && w.wq_fp4_sf && w.wk_fp4 && w.wk_fp4_sf &&
                    w.wv_fp4 && w.wv_fp4_sf && fp4_attn_a && fp4_attn_as && fp4_attn_ws &&
                    kernels::launch_prefill_nvfp4_quant_a(xn, fp4_attn_a, fp4_attn_as, N, H, st) &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_attn_a, fp4_attn_as,
                                                       w.wq_fp4, w.wq_fp4_sf,
                                                       b8, N, wide, H, fp4_attn_ws, st,
                                                       w.wq_fp4_alpha) &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_attn_a, fp4_attn_as,
                                                       w.wk_fp4, w.wk_fp4_sf,
                                                       kf, N, kvdim, H, fp4_attn_ws, st,
                                                       w.wk_fp4_alpha) &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_attn_a, fp4_attn_as,
                                                       w.wv_fp4, w.wv_fp4_sf,
                                                       vf, N, kvdim, H, fp4_attn_ws, st,
                                                       w.wv_fp4_alpha))
                    qkv_fp4 = true;
                bool grouped = qkv_fp4;
                if (!qkv_fp4 && use_i8 && w.wq_rs && w.wk_rs && w.wv_rs &&
                    w.wk_type == w.wq_type && w.wv_type == w.wq_type &&
                    kernels::pf_dense_gemm_qi8_supported(w.wq_type)) {
                    const void*  Wa[3]  = { w.wq, w.wk, w.wv };
                    const float* rsa[3] = { w.wq_rs, w.wk_rs, w.wv_rs };
                    void*        Ca[3]  = { b8, kf, vf };
                    const int    na[3]  = { wide, kvdim, kvdim };
                    quant_a_i8(xn, N, H);
                    grouped = kernels::launch_prefill_gemm_qi8_dense_group(
                        w.wq_type, A_i8, sx, Wa, rsa, Ca, na, 3, N, H, st,
                        qb_partials, QB_SPLITS, qb_partials_cap,
                        nullptr, nullptr, nullptr, apk());
                }
                if (!grouped) {
                    proj_fused(xn, w.wq, w.wq_type, w.wq_rs, b8, wide,  H);  // qraw = [q|gate]
                    proj_fused(xn, w.wk, w.wk_type, w.wk_rs, kf, kvdim, H);
                    proj_fused(xn, w.wv, w.wv_type, w.wv_rs, vf, kvdim, H);
                }
                kernels::launch_prefill_split_q_gate(b8, qb, qg, N, c.n_q_heads, c.head_dim, st);
            }
            if (c.muse_glimmer) {
                // Muse Glimmer runs a BF16 KV cache (its decode KV-write is bf16, qwen35.cpp:1065/1087),
                // not int8. QK-norm + NORMAL (consecutive-pair, LLAMA_ROPE_TYPE_NORM) RoPE on SWA layers
                // / NoPE on global layers, bf16 append -- the same KV a forward_token decode writes via
                // launch_rmsnorm + launch_rope_kv_append_normal / launch_kv_append. Then pure-window
                // attention on SWA layers, full causal on global (win_blocks<=0), over the bf16 pool.
                bf16* kpool_bf = (bf16*)s.kv->k_pool() + s.kv->layer_base_elems(L);
                bf16* vpool_bf = (bf16*)s.kv->v_pool() + s.kv->layer_base_elems(L);
                const int muse_rot = w.swa ? c.head_dim : 0;      // SWA = full NORMAL rope; global = NoPE
                kernels::launch_prefill_qknorm_ropenorm_kv_bf16(qb, kf, vf, w.q_norm, w.k_norm,
                    kpool_bf, vpool_bf, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                    muse_rot, rope_theta, eps, bs, mbs, st);
                const int win_blocks = w.swa ? (c.sliding_window + bs - 1) / bs : 0;  // 0 => global full causal
                kernels::launch_prefill_attn_swa_pure_bf16(qb, kpool_bf, vpool_bf, btable, att,
                    N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale, win_blocks, st);
            } else {
                signed char* kpool = (signed char*)s.kv->k_pool() + s.kv->layer_base_elems(L) * kv_elem;
                signed char* vpool = (signed char*)s.kv->v_pool() + s.kv->layer_base_elems(L) * kv_elem;
                void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + s.kv->scale_layer_base_elems(L) * 2 : nullptr;
                void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + s.kv->scale_layer_base_elems(L) * 2 : nullptr;
                // bf16 KV: batched prefill used to decline here, which sent Qwen3.8's prefill@128
                // down the sequential per-token path (88.0 tok/s, barely above its own 84.2 tok/s
                // decode, because that path re-streams every weight once per position). The bf16
                // twins below take the same schedule as the int8 pair with the quantize removed,
                // and write exactly the bf16 KV a forward_token decode writes when kv8 is off, so
                // a decode continuing from this cache reads a consistent one.
                // SPARKINFER_PREFILL_BF16_KV=0 restores the decline.
                static int pf_bf16kv = -1;
                if (pf_bf16kv < 0) { const char* e = getenv("SPARKINFER_PREFILL_BF16_KV"); pf_bf16kv = (e && e[0] == '0') ? 0 : 1; }
                if (!kv8) {
                    // Capability check by shape, NOT by calling the launcher -- invoking it as a
                    // probe would really launch the kernel, on the wrong stream and over a KV pool
                    // this pass has not written yet.
                    const bool bf16_attn_ok = pf_bf16kv && (c.head_dim == 128 || c.head_dim == 256);
                    if (!bf16_attn_ok) {
                        a.free_all(); a8.free_all(); am.free_all(); aw.free_all();
                        fprintf(stderr, "[prefill] batched prefill requires int8 KV\n");
                        return -1;
                    }
                    kernels::launch_prefill_qknorm_rope_kv_bf16(qb, kf, vf, w.q_norm, w.k_norm,
                        kpool, vpool, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                        rope_dim, rope_theta, eps, bs, mbs, st);
                    kernels::launch_prefill_attn_bf16_paged(qb, kpool, vpool, btable, att,
                        N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale, st);
                } else {
                    kernels::launch_prefill_qknorm_rope_kv_int8(qb, kf, vf, w.q_norm, w.k_norm,
                        kpool, vpool, kscale, vscale, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                        rope_dim, rope_theta, eps, bs, mbs, st);
                    kernels::launch_prefill_attn_int8_paged(qb, kpool, vpool, kscale, vscale, btable, att,
                        N, c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale, st);
                }
            }
            // Muse: the gated attention output feeds exactly one consumer -- the o projection's
            // row-quantize -- so fold the gate into that quantize's load phase. `att` is then never
            // written back as bf16 and never re-read, and one launch per layer goes away.
            // Bit-identical; only taken when the o projection is certain to use A_i8 (otherwise
            // proj() would read the un-gated `att`).
            //
            // The block-scaled o projection folds the gate into its OWN quantize, so it is tried
            // first and the int8 fold is skipped when it succeeds. Every arm therefore consumes a
            // gated activation and nothing reads the raw `att` -- the invariant #816 broke.
            const bool wo_fp4_gated = muse_nvfp4_wo && w.wo_fp4 && w.wo_fp4_sf && fp4_a && fp4_as &&
                kernels::launch_prefill_nvfp4_gate_quant_a(att, qg, fp4_a, fp4_as, N, qdim, st);
            const bool wo_fp4_done = wo_fp4_gated && c.muse_glimmer &&
                kernels::launch_prefill_nvfp4_gemm(fp4_a, fp4_as, w.wo_fp4, w.wo_fp4_sf,
                                                   ao, N, H, qdim, fp4_ws, st);
            bool gate_fused = false;
            if (!wo_fp4_gated &&
                c.muse_glimmer && muse_qb && use_i8 && w.wo_rs && H >= 128 &&
                kernels::pf_dense_gemm_qi8_supported(w.wo_type) &&
                kernels::launch_prefill_gate_quant_rows_i8(att, qg, A_i8, sx, N, qdim, st,
                                                           A_i8p)) {
                a_q = att; a_qR = N; a_qK = qdim;      // quant_a_i8(att, N, qdim) is now a no-op
                a_pk = A_i8p != nullptr;
                gate_fused = true;
            }
            // If the fused quantize ran but the GEMM declined, `att` is still raw -- gate it here.
            if (!gate_fused && !wo_fp4_done) {
                kernels::launch_prefill_mul_sigmoid(att, qg, N, qdim, st);
            }
            // o off the same NVFP4 bytes, reading the already-gated `att`, with the residual taken
            // by the epilogue's C operand (same fold as the GDN out_proj above) so no separate
            // full-tensor add pass is needed. If the fused form declines, fall back to writing the
            // raw projection to `ao` and let the `if (!attn_fused) launch_prefill_add(...)` tail
            // apply it, exactly as the Muse wo_fp4 arm does.
            bool wo_fp4_q38 = false, wo_fp4_resid = false;
            if (attn_nvfp4 && (attn_fp4_mask & 2) &&
                w.wo_fp4 && w.wo_fp4_sf && fp4_attn_a && fp4_attn_as && fp4_attn_ws &&
                kernels::launch_prefill_nvfp4_quant_a(att, fp4_attn_a, fp4_attn_as, N, qdim, st)) {
                if (nvfp4_resid_fuse && !c.muse_glimmer &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_attn_a, fp4_attn_as,
                                                       w.wo_fp4, w.wo_fp4_sf,
                                                       x, N, H, qdim, fp4_attn_ws, st,
                                                       w.wo_fp4_alpha, x)) {
                    wo_fp4_q38 = true;
                    wo_fp4_resid = true;
                } else if (kernels::launch_prefill_nvfp4_gemm(fp4_attn_a, fp4_attn_as,
                                                              w.wo_fp4, w.wo_fp4_sf,
                                                              ao, N, H, qdim, fp4_attn_ws, st,
                                                              w.wo_fp4_alpha)) {
                    wo_fp4_q38 = true;
                }
            }
            if (wo_fp4_q38) {
                attn_fused = wo_fp4_resid;
            } else if (c.muse_glimmer) {
                // Sandwich norm needs the RAW O-proj output in `ao` (not fused into x); the residual
                // add happens in launch_norm_then_add below. The FP4 GEMM already wrote `ao` and
                // left attn_acc at 0, so the plain norm_then_add tail consumes it.
                if (!wo_fp4_done)
                    proj_fused_acc(att, w.wo, w.wo_type, w.wo_rs, ao, H, qdim, &attn_acc);
                attn_fused = false;
            } else if (w.wo_rs) {
                proj_fused(att, w.wo, w.wo_type, w.wo_rs, ao, H, qdim);
                attn_fused = false;
            } else {
                attn_fused = proj_resid(att, w.wo, w.wo_type, x, H, qdim);
                if (!attn_fused) proj(att, w.wo, w.wo_type, ao, H, qdim);
            }
            use_i8 = restore_i8;
        }

        if (c.muse_glimmer) {
            // Sandwich norm (post-attn): h = x + RMSNorm(ao) * post_attn_norm -- norm the attention
            // output ALONE, then add to the residual (decode qwen35.cpp:1112). ffn_norm is a genuine
            // SEPARATE pre-FFN norm here, not post_attn_norm doing double duty like every other arch.
            // Sandwich (post_attn_norm/post_ffn_norm) RMSNorm uses its OWN eps 1e-8
            // (upstream post_norm_eps), NOT the model's rms_eps (1e-5) which drives
            // attn_norm/ffn_norm/q_norm/k_norm -- mirrors the decode fix (qwen35.cpp:6d911d4).
            if (attn_acc)
                kernels::launch_norm_then_add_acc(x, qb_partials, sx, w.wo_rs, w.post_attn_norm,
                                                  h, N, H, 1e-8f, st);
            else
                kernels::launch_norm_then_add(x, ao, w.post_attn_norm, h, N, H, 1e-8f, st);
            // hn's only consumer is the grouped FFN's row-quantize, so emit the int8 in the same
            // pass. Only when one chunk covers the prompt: a second chunk would need A_i8/sx again
            // after the first has overwritten them. The bf16 hn is still written either way.
            if (muse_ffn_group && muse_qb && use_i8 && FC >= N &&
                kernels::launch_rmsnorm_quant_i8(h, w.ffn_norm, hn, A_i8, sx, N, H, eps, st,
                                                 A_i8p)) {
                hn_quantized = true;
                // This kernel writes A_i8 itself, so it owns the k-tiled copy's validity too --
                // the memo below makes quant_a_i8 a no-op, which would otherwise leave a_pk
                // asserting the ATTENTION activation still packed at cols=qdim.
                a_pk = A_i8p != nullptr;
            } else {
                kernels::launch_rmsnorm(h, w.ffn_norm, hn, N, H, eps, st);
            }
        } else {
            // x += ao (post-attn residual, in-place; skipped when folded into the output proj)
            // hn = RMSNorm(x, post_attn_norm)
            if (!attn_fused) kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
            kernels::launch_rmsnorm(x, w.post_attn_norm, hn, N, H, eps, st);
        }

        if (!moe) {
            // dense SwiGLU FFN, chunked over tokens (upstream #530): ffg/ffu/A_i8 stay O(FC*ffn).
            const void* gate_pf = w.prefill_gate_q ? w.prefill_gate_q : w.gate_q;
            const void* up_pf = w.prefill_up_q ? w.prefill_up_q : w.up_q;
            const int gate_pf_type = w.prefill_gate_q ? w.prefill_gate_qtype : w.gate_qtype;
            const int up_pf_type = w.prefill_up_q ? w.prefill_up_qtype : w.up_qtype;
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
            const bool ffn_qi8 = use_i8 && w.gate_rs && w.up_rs &&
                kernels::pf_dense_gemm_qi8_supported(gate_pf_type);
            if (ffn_i8 && !ffn_qi8) {
                dequant_w_i8(gate_pf_type, gate_pf, ffn_Wg_i8, ffn_swg, ffn, H);
                dequant_w_i8(up_pf_type,   up_pf,   ffn_Wu_i8, ffn_swu, ffn, H);
                dequant_w_i8(w.down_qtype, w.down_q, ffn_Wd_i8, ffn_swd, H, ffn);
            }
            // The down projection takes the int8 path on both branches whenever ffn_i8 or use_i8,
            // so the FFN residual can ride the residual-fused GEMM straight into x per chunk.
            // Muse Glimmer must NOT residual-fuse: the post-FFN sandwich (norm_then_add below) needs
            // the RAW FFN output in `ao`, and the residual it adds onto is `h` (not x).
            // The FP4 arm below writes the RAW FFN output to `ao` and never folds the residual --
            // it ends in `continue`, skipping every fused-residual path in this loop. So the fused
            // decision has to account for it, or the post-loop
            //     } else if (!ffn_fused) { x += ao; }
            // is skipped on the belief the down GEMM already accumulated into x, and the whole FFN
            // contribution is dropped from the residual stream, on every one of 64 layers, for
            // every token of every prefill. Fluent output, wrong content, worse with context.
            //
            // Muse Glimmer shares the same FP4 arm and was never affected: c.muse_glimmer already
            // forces ffn_fused false, and its sandwich norm consumes `ao` explicitly, so its FP4
            // output was always read back. That asymmetry is why this looked like a Qwen3.8 kernel
            // bug -- the NVFP4 GEMM is exact to ~0.14 rel err at both models' shapes
            // (runtime/examples/nvfp4_gemm_check.cpp); the residual bookkeeping around it was not.
            //
            // Keyed on whether the arm CAN run, not on whether a given chunk took it: chunks would
            // otherwise mix fused and unfused within one layer and a single post-loop add could not
            // be right for both. With this false the non-FP4 chunks write `ao` too, so one add at
            // the end covers every chunk.
            const bool ffn_fp4_possible = gu_nvfp4 && w.gate_fp4 && w.gate_fp4_sf &&
                                          w.up_fp4 && w.up_fp4_sf && fp4_a && fp4_as;
            const bool ffn_fused = !c.muse_glimmer && !ffn_fp4_possible &&
                                   resid_fuse && (ffn_i8 || use_i8) && !ffn_qi8;
            // The FP4 down projection accumulates the residual in its own epilogue (see the GDN
            // out_proj above). Decided per LAYER, not per chunk, so the whole layer takes one
            // route: a chunk that fuses has already applied its residual to x, so a mix would
            // need the trailing add to skip exactly those rows. When it is on, any chunk whose
            // fused GEMM declines applies its own residual immediately instead.
            const bool ffn_fp4_resid = nvfp4_resid_fuse && ffn_fp4_possible && !c.muse_glimmer &&
                                       nvfp4_down && w.down_fp4 && w.down_fp4_sf &&
                                       fp4_down_a && fp4_down_as;
            for (int fo = 0; fo < N; fo += FC) {
                const int fn = (N - fo < FC) ? (N - fo) : FC;
                const bf16* hn_c = hn + (size_t)fo * H;
                const bool layer_fp4 = gu_nvfp4 && w.gate_fp4 && w.gate_fp4_sf &&
                    w.up_fp4 && w.up_fp4_sf && fp4_a && fp4_as &&
                    kernels::prefill_nvfp4_supported(fn, ffn, H) &&
                    kernels::launch_prefill_nvfp4_quant_a(hn_c, fp4_a, fp4_as, fn, H, st) &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_a, fp4_as, w.gate_fp4, w.gate_fp4_sf,
                                                       ffg, fn, ffn, H, fp4_ws, st,
                                                       w.gate_fp4_alpha) &&
                    kernels::launch_prefill_nvfp4_gemm(fp4_a, fp4_as, w.up_fp4, w.up_fp4_sf,
                                                       ffu, fn, ffn, H, fp4_ws, st,
                                                       w.up_fp4_alpha);
                if (layer_fp4) {
                    bf16* xc = x + (size_t)fo * H;
                    const bool down_swiglu_q = nvfp4_down && w.down_fp4 && w.down_fp4_sf &&
                        fp4_down_a && fp4_down_as &&
                        kernels::launch_prefill_nvfp4_swiglu_quant_a(
                            ffg, ffu, fp4_down_a, fp4_down_as, fn, ffn, st);
                    const bool down_fp4_resid = down_swiglu_q && ffn_fp4_resid &&
                        kernels::launch_prefill_nvfp4_gemm(
                            fp4_down_a, fp4_down_as, w.down_fp4, w.down_fp4_sf,
                            xc, fn, H, ffn, fp4_ws, st, w.down_fp4_alpha, xc);
                    const bool down_fp4_done = down_fp4_resid ||
                        (down_swiglu_q &&
                         kernels::launch_prefill_nvfp4_gemm(
                            fp4_down_a, fp4_down_as, w.down_fp4, w.down_fp4_sf,
                            ao + (size_t)fo * H, fn, H, ffn, fp4_ws, st,
                            w.down_fp4_alpha));
                    if (!down_fp4_done) {
                        kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                        proj_fused_acc(ffg, w.down_q, w.down_qtype, w.down_rs,
                                       ao + (size_t)fo * H, H, ffn, &ffn_acc, fn);
                    }
                    // Keep the layer on one route: if the residual-fused arm is active but this
                    // chunk fell through it, that chunk's rows are still raw in `ao` and the
                    // trailing whole-tensor add is going to be skipped, so apply them here.
                    if (ffn_fp4_resid && !down_fp4_resid)
                        kernels::launch_prefill_add(xc, ao + (size_t)fo * H, xc, (long)fn * H, st);
                    continue;
                }
                if (ffn_qi8) {
                    bool gu_grouped = false;
                    if (gate_pf_type == up_pf_type && w.gate_rs && w.up_rs) {
                        const void*  Wf[2]  = { gate_pf, up_pf };
                        const float* rsf[2] = { w.gate_rs, w.up_rs };
                        void*        Cf[2]  = { ffg, ffu };
                        const int    nf[2]  = { ffn, ffn };
                        quant_a_i8(hn_c, fn, H);
                        gu_grouped = kernels::launch_prefill_gemm_qi8_dense_group(
                            gate_pf_type, A_i8, sx, Wf, rsf, Cf, nf, 2, fn, H, st,
                            qb_partials, QB_SPLITS, qb_partials_cap,
                            nullptr, nullptr, nullptr, apk());
                    }
                    if (!gu_grouped) {
                        proj_fused(hn_c, gate_pf, gate_pf_type, w.gate_rs, ffg, ffn, H, fn);
                        proj_fused(hn_c, up_pf,   up_pf_type,   w.up_rs,   ffu, ffn, H, fn);
                    }
                    a_q = nullptr;
                    a_pk = kernels::launch_prefill_swiglu_quant_i8(ffg, ffu, A_i8, sx, fn, ffn, st,
                                                                   A_i8p) && A_i8p;
                    bool down_fused = false;
                    if (w.down_rs && kernels::pf_dense_gemm_qi8_supported(w.down_qtype)) {
                        down_fused = kernels::launch_prefill_gemm_qi8_dense(
                            w.down_qtype, A_i8, sx, w.down_q, w.down_rs,
                            ao + (size_t)fo * H, fn, H, ffn, st,
                            qb_partials, QB_SPLITS, nullptr, apk());
                    }
                    if (!down_fused) {
                        if (!kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q,
                                                                  W_i8, sw, H, ffn, st)) {
                            const void* wb = dq(w.down_q, w.down_qtype, H, ffn);
                            kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, H, ffn, st);
                        }
                        if (ffn_fused)
                            gemm_i8(A_i8, W_i8, sx, sw, x + (size_t)fo * H, fn, H, ffn, true);
                        else
                            gemm_i8(A_i8, W_i8, sx, sw, ao + (size_t)fo * H, fn, H, ffn, false);
                    }
                    continue;
                }
                if (ffn_i8) {
                    a_q = nullptr;                     // this branch writes A_i8/sx directly
                    a_pk = kernels::launch_prefill_quantize_rows_i8(hn_c, A_i8, sx, fn, H, st,
                                                                    A_i8p) && A_i8p;
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wg_i8, sx, ffn_swg, ffg, fn, ffn, H, st);
                    kernels::launch_prefill_gemm_i8(A_i8, ffn_Wu_i8, sx, ffn_swu, ffu, fn, ffn, H, st);
                    // fused SwiGLU + int8 quantize for the down input (skips the ffg DRAM round-trip)
                    a_pk = kernels::launch_prefill_swiglu_quant_i8(ffg, ffu, A_i8, sx, fn, ffn, st,
                                                                   A_i8p) && A_i8p;
                    if (ffn_fused)
                        kernels::launch_prefill_gemm_i8_resid(A_i8, ffn_Wd_i8, sx, ffn_swd,
                                                              x + (size_t)fo * H, fn, H, ffn, st);
                    else
                        kernels::launch_prefill_gemm_i8(A_i8, ffn_Wd_i8, sx, ffn_swd,
                                                        ao + (size_t)fo * H, fn, H, ffn, st);
                } else {
                    // Group the retained native prefill weights; decode-only Q3_A buffers are not
                    // supported by this fused Q4_K/Q5_K prefill kernel.
                    bool ffn_grouped = false;
                    // Set by the grouped launcher when it folded the SwiGLU + int8 quantize into
                    // its split-K epilogue, so gate/up were never written out as bf16.
                    int ffn_fused_swiglu = 0;
                    if (muse_ffn_group && muse_qb && use_i8 && w.gate_rs && w.up_rs &&
                        gate_pf_type == up_pf_type &&
                        kernels::pf_dense_gemm_qi8_supported(gate_pf_type)) {
                        const void*  Wf[2]  = { gate_pf, up_pf };
                        const float* rsf[2] = { w.gate_rs, w.up_rs };
                        void*        Cf[2]  = { ffg, ffu };
                        const int    nf[2]  = { ffn, ffn };
                        if (hn_quantized) { a_q = hn_c; a_qR = fn; a_qK = H; }   // already done
                        quant_a_i8(hn_c, fn, H);
                        // A_i8/sx are handed in as the fused SwiGLU's OUTPUT as well as the GEMM's
                        // input: the GEMM has completed in stream order before the epilogue runs, and
                        // each epilogue block reads sx for its own row into a register before writing
                        // it back, so the aliasing is safe -- and it is exactly what the unfused
                        // launch_prefill_swiglu_quant_i8 call below already does.
                        ffn_grouped = kernels::launch_prefill_gemm_qi8_dense_group(
                            gate_pf_type, A_i8, sx, Wf, rsf, Cf, nf, 2, fn, H, st,
                            qb_partials, QB_SPLITS, qb_partials_cap,
                            A_i8, sx, &ffn_fused_swiglu, apk(), A_i8p);
                    }
                    if (!ffn_grouped) {
                        proj_fused(hn_c, gate_pf, gate_pf_type, w.gate_rs, ffg, ffn, H, fn);
                        proj_fused(hn_c, up_pf,   up_pf_type,   w.up_rs,   ffu, ffn, H, fn);
                    }
                    if (use_i8) {
                        // Same fused SwiGLU + per-row int8 quantize the long-ctx ffn_i8 branch
                        // runs (bit-identical to swiglu-then-quantize; both bf16-round first) --
                        // skips the ffg store + reload that proj()'s internal quantize would pay.
                        a_q = nullptr;                 // swiglu_quant writes A_i8/sx directly
                        // The fused epilogue already emitted the k-tiled copy (fuse_qp below).
                        if (!ffn_fused_swiglu)
                            a_pk = kernels::launch_prefill_swiglu_quant_i8(ffg, ffu, A_i8, sx, fn,
                                                                          ffn, st, A_i8p) && A_i8p;
                        else
                            a_pk = A_i8p != nullptr;
                        // Fused quantized-B down projection. The activation is ALREADY in A_i8/sx
                        // (the fused SwiGLU wrote it), so this cannot go through proj_fused, which
                        // would re-quantize -- call the dense fused GEMM directly. Only the
                        // residual-unfused form is eligible; Muse never takes the fused-residual
                        // branch anyway (its sandwich norm needs the raw FFN output in ao).
                        bool down_fused = false;
                        if (!ffn_fused && muse_qb && w.down_rs &&
                            kernels::pf_dense_gemm_qi8_supported(w.down_qtype)) {
                            // The accumulator can only stand when this chunk IS the whole prompt:
                            // a second chunk would overwrite both qb_partials and sx before the
                            // post-FFN norm below reads them.
                            down_fused = kernels::launch_prefill_gemm_qi8_dense(
                                w.down_qtype, A_i8, sx, w.down_q, w.down_rs,
                                ao + (size_t)fo * H, fn, H, ffn, st, qb_partials, QB_SPLITS,
                                (fn == N) ? &ffn_acc : nullptr, apk());
                        }
                        if (!down_fused) {
                            if (!kernels::launch_gguf_dequant_rows_i8(w.down_qtype, w.down_q,
                                                                      W_i8, sw, H, ffn, st)) {
                                const void* wb = dq(w.down_q, w.down_qtype, H, ffn);
                                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, H, ffn, st);
                            }
                            // keeps #795's split-K fan-out for whatever still materializes (Q6_K down)
                            if (ffn_fused)
                                gemm_i8(A_i8, W_i8, sx, sw, x + (size_t)fo * H, fn, H, ffn, true);
                            else
                                gemm_i8(A_i8, W_i8, sx, sw, ao + (size_t)fo * H, fn, H, ffn, false);
                        }
                    } else {
                        kernels::launch_prefill_swiglu(ffg, ffu, ffg, (long)fn * ffn, st);
                        if (!ffn_fused || !proj_resid(ffg, w.down_q, w.down_qtype,
                                                      x + (size_t)fo * H, H, ffn, fn))
                            proj(ffg, w.down_q, w.down_qtype, ao + (size_t)fo * H, H, ffn, fn);
                    }
                }
            }
            if (c.muse_glimmer) {
                // Sandwich norm (post-FFN): x = h + RMSNorm(ao) * post_ffn_norm (decode
                // qwen35.cpp:1329). h is the post-attn residual stream; ao holds the raw FFN output.
                // Same 1e-8 post_norm_eps as the post-attn sandwich norm above.
                if (ffn_acc)
                    kernels::launch_norm_then_add_acc(h, qb_partials, sx, w.down_rs,
                                                      w.post_ffn_norm, x, N, H, 1e-8f, st);
                else
                    kernels::launch_norm_then_add(h, ao, w.post_ffn_norm, x, N, H, 1e-8f, st);
            } else if (!ffn_fused && !ffn_fp4_resid) {
                // x += ffn_out (skipped when the down GEMM already accumulated into x per chunk,
                // whether through the int8 fused-residual GEMM or the FP4 epilogue's C operand)
                kernels::launch_prefill_add(x, ao, x, (long)N * H, st);
            }
        } else {
            // ---- expert-grouped 256-expert int8 MoE FFN: route -> bucket routed
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
    // Q4_K head: quantize the activation ONCE and run the pre-quantized dp4a GEMV. gemv.cu calls
    // this BIT-EXACT vs the in-kernel path -- same Q8_1 values, same dp4a -- and it drops the
    // per-block re-quantization of the same 6656-value activation, which at vocab-many rows is the
    // larger half of what that one launch reads after the weights themselves.
    if (s.w.lm_head_type == 12 && lm_q8 && lm_ad && lm_as) {
        kernels::launch_quantize_q8_1(xn_last, lm_q8, lm_ad, lm_as, H, st);
        kernels::launch_gemv_q_dp4a_pq_f32(lm_q8, lm_ad, lm_as, s.w.lm_head, s.logits, c.vocab, H, st);
    } else if (s.w.lm_head_type)
        kernels::launch_gemv_q_f32(xn_last, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else
        kernels::launch_gemv_f32(xn_last, s.w.lm_head, s.logits, c.vocab, H, st);
    // Muse Glimmer tanh final-logit softcap before argmax (decode qwen35.cpp:1365).
    if (c.muse_glimmer && c.final_logit_softcapping > 0.f)
        kernels::launch_logit_softcap(s.logits, 1, c.vocab, c.logit_scale, c.final_logit_softcapping, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    // Close the capture BEFORE the D2H + sync: a synchronize cannot be recorded, and the seed
    // readback is per-call anyway. Capturing records without executing, so the graph is launched
    // here to actually produce THIS call's result.
    if (pfb_capturing) {
        cudaGraph_t g = nullptr;
        if (cudaStreamEndCapture(st, &g) == cudaSuccess && g) {
            cudaGraphExec_t e = nullptr;
            if (cudaGraphInstantiate(&e, g, nullptr, nullptr, 0) == cudaSuccess) {
                if (g_pfb_graph) cudaGraphDestroy(g_pfb_graph);
                g_pfb_graph = g; g_pfb_exec = e; g_pfb_n = N;
                pf_cu(cudaGraphLaunch(e, st), "pfb first launch");
            } else {
                // Nothing ran (capture records, it does not execute) and there is no graph to run
                // it with. Report failure so the caller falls back to the token loop rather than
                // returning a seed from uninitialised memory.
                cudaGraphDestroy(g);
                fprintf(stderr, "[prefill] graph instantiate failed -> fallback\n");
                return -1;
            }
        } else {
            fprintf(stderr, "[prefill] graph capture failed -> fallback\n");
            return -1;
        }
    }
    g_pfb_warm_n = N;
    pf_cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "prefill seed");
    pf_cu(cudaStreamSynchronize(st), "prefill sync");
    int seed = *s.h_out_id;

    // Release rather than hold when this call's scratch is too big to keep resident.
    if (!arena_reuse ||
        a.total() + a8.total() + am.total() + aw.total() > kArenaKeepBytes) {
        a.free_all();
        a8.free_all();
        am.free_all();
        aw.free_all();
        // Any captured graph (g_pfb_exec) has these just-freed arena addresses baked into its
        // kernel launch params -- replaying it via the fast path at the top of this function
        // (cudaGraphLaunch against a stale g_pfb_n match) would touch freed device memory and
        // segfault (#809: reproduced when a prefill N large enough to blow the keep-resident
        // budget was captured, then replayed 1+ more times after this cleanup ran). Tear the
        // graph down here too and reset g_pfb_warm_n so the next call redoes the warm-then-
        // capture cycle against fresh (post-free) addresses rather than capturing over a cold
        // allocation.
        if (g_pfb_exec)  { cudaGraphExecDestroy(g_pfb_exec); g_pfb_exec = nullptr; }
        if (g_pfb_graph) { cudaGraphDestroy(g_pfb_graph); g_pfb_graph = nullptr; }
        g_pfb_n = -1;
        g_pfb_warm_n = -1;
    }
    return seed;
}

int dflash_verify_short_run(const Qwen35PrefillCtx& s, const int* token_ids, int n, int start_pos,
                            const int* capture_layers, int n_capture, void* capture_dst,
                            int* out_argmax, bool capture_only) {
    const Qwen35Config& c = s.cfg;
    // dense_ffn (Qwen3.8-27B) is accepted alongside the 256-expert MoE (Qwen3.6-35B-A3B) it was
    // written for. A dense SwiGLU is an MoE with ONE expert: AR decode already routes it through
    // launch_moe_expert_ffn_q4k with a constant expert id 0 and weight 1.0 (qwen35.cpp's
    // c.dense_ffn branch, and the mf_ids/mf_weights seeding at model construction), so the
    // verifier's FFN stage needs the router skipped and those constants supplied -- not a separate
    // dense implementation.
    //
    // This is a CORRECTNESS prerequisite, not a throughput one: with these guards rejecting the
    // model, dflash_generate falls back to a token-loop path that is not lossless here (it emits
    // unverified draft tokens -- observed echoing the prompt back), so DSpark cannot work at all
    // until this branch exists.
    const bool dense = c.dense_ffn;
    if (!token_ids || !out_argmax || n < 1 || n > 8 || !s.gguf || !c.hybrid) {
        fprintf(stderr, "[dflash-verify] base unsupported n=%d gguf=%d hybrid=%d\n",
                n, (int)s.gguf, (int)c.hybrid);
        return -1;
    }
    if (c.head_dim != 256 || c.linear_head_dim != 128 || c.top_k <= 0 ||
        (!dense && c.n_experts != 256)) {
        fprintf(stderr, "[dflash-verify] shape unsupported hd=%d lhd=%d experts=%d topk=%d dense=%d\n",
                c.head_dim, c.linear_head_dim, c.n_experts, c.top_k, (int)dense);
        return -1;
    }
    const int H = c.hidden, N = n, qdim = s.qdim, kvdim = s.kvdim;
    const int lqkv = s.linear_qkvdim, lvdim = s.linear_vdim, vh = c.linear_v_heads;
    const int ffn = c.moe_ffn, E = c.n_experts, topk = c.top_k;
    // DEBUG ONLY (dspark_tau_check bisection, 2026-08-17): confirm the row<->position mapping
    // before trusting any hidden-state diff. Real token ids only (warm_verify's capture_only
    // pre-build pass uses all-zero placeholder ids, not real ones -- skip printing those).
    if (getenv("SPARKINFER_DFLASH_VERIFY_DUMP_ROW") && token_ids[0] != 0) {
        fprintf(stderr, "[dflash-verify-debug] start_pos=%d N=%d ids=[", start_pos, N);
        for (int i = 0; i < N; i++) fprintf(stderr, "%d ", token_ids[i]);
        fprintf(stderr, "]\n");
    }
    cudaStream_t st = s.stream;
    static thread_local Arena verify_arena;
    verify_arena.rewind();
    Arena& a = verify_arena;
    bf16* x = a.alloc<bf16>((size_t)N * H);
    bf16* xn = a.alloc<bf16>((size_t)N * H);
    bf16* h = a.alloc<bf16>((size_t)N * H);
    // DEBUG ONLY (dspark_tau_check bisection, 2026-08-17): SPARKINFER_DFLASH_VERIFY_DUMP_ROW=<row>
    // dumps that row's pre-attn-norm xn after EVERY layer, plus the post-final-norm xn, into
    // [n_layers+1, H] bf16 written to SPARKINFER_DFLASH_VERIFY_DUMP_FILE (default
    // /tmp/dflash_verify_xn_dump.bin) -- same format as qwen35.cpp's SPARKINFER_MG_DUMP_STEP, so
    // the two can be diffed layer-by-layer to find where a verify row's hidden state first parts
    // ways from AR's. SPARKINFER_DFLASH_VERIFY_DUMP_START_POS=<pos> (default: first call, any
    // start_pos) selects WHICH verify round to dump -- rows are the DRAFT's own proposed tokens,
    // not a 1:1 window onto the AR reference sequence, so "first call" only lines up with AR's
    // first few outputs when every earlier row in that round was also a correct guess.
    static int verify_dbg_row = -2;
    if (verify_dbg_row < -1) {
        const char* e = getenv("SPARKINFER_DFLASH_VERIFY_DUMP_ROW");
        verify_dbg_row = e ? atoi(e) : -1;
    }
    static int verify_dbg_want_pos = -2;
    if (verify_dbg_want_pos < -1) {
        const char* e = getenv("SPARKINFER_DFLASH_VERIFY_DUMP_START_POS");
        verify_dbg_want_pos = e ? atoi(e) : -1;
    }
    static int verify_dbg_call = 0;
    static bf16* verify_dbg_buf = nullptr;
    // Two separate gates, deliberately not one: whichever call happens to be first triggers this
    // function's ONE graph recording (dflash_warm_verify's capture_only pre-build, typically at
    // an earlier start_pos than the round actually wanted). A CUDA graph replay does NOT re-run
    // this function's C++ -- it only relaunches whatever kernel nodes were captured during that
    // first recording -- so gating the memcpy itself on start_pos meant it was never captured at
    // all when the target round wasn't the recording one, and every "dump" after that silently
    // read uninitialized verify_dbg_buf. Snapshot on every call (cheap, and bakes the node into
    // the graph regardless of which call records it); gate only the FILE WRITE on start_pos, and
    // since replay still feeds live data through the captured nodes, the buffer is guaranteed
    // correct for the CURRENT call by the time verify sync/replay-sync completes.
    const bool vdbg = (verify_dbg_row >= 0 && verify_dbg_row < N);
    const bool vdbg_dump_now = (vdbg && verify_dbg_call == 0 &&
                                (verify_dbg_want_pos < 0 || verify_dbg_want_pos == start_pos));
    if (vdbg && !verify_dbg_buf)
        pf_cu(cudaMalloc(&verify_dbg_buf, (size_t)(c.n_layers + 1) * H * sizeof(bf16)), "verify_dbg_buf alloc");
    auto vdbg_snapshot = [&](const bf16* p, int layer) {
        if (vdbg) pf_cu(cudaMemcpyAsync(verify_dbg_buf + (size_t)layer * H, p + (size_t)verify_dbg_row * H,
                                        (size_t)H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
                        "verify_dbg snapshot");
    };
    // DEBUG ONLY: 4-slot layer-0-only sub-stage buffer (ao/h/hn/routed) -- see the call sites.
    static bf16* verify_dbg_buf2 = nullptr;
    if (vdbg && !verify_dbg_buf2)
        pf_cu(cudaMalloc(&verify_dbg_buf2, (size_t)5 * H * sizeof(bf16)), "verify_dbg_buf2 alloc");
    auto vdbg_snapshot2 = [&](const bf16* p, int slot) {
        if (vdbg) pf_cu(cudaMemcpyAsync(verify_dbg_buf2 + (size_t)slot * H, p + (size_t)verify_dbg_row * H,
                                        (size_t)H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
                        "verify_dbg2 snapshot");
    };
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
    // out_scratch is not just the [N, H] fp32 output: launch_moe_expert_ffn_q4k also reuses it as
    // the Q8_1 staging buffer for the SwiGLU hidden, which needs
    // num_tokens * top_k * llama_q8_1_bytes(ffn) bytes. That kernel's "<= hidden floats; fits"
    // reasoning holds only for num_tokens == 1; the batched verify calls it with N rows and
    // overruns an [N, H] float buffer whenever 9*ffn > 4*H, silently corrupting the rows staged
    // after the overflow point. Size for both uses.
    const size_t moe_out_floats = std::max((size_t)N * H,
        ((size_t)N * topk * kernels::llama_q8_1_bytes(ffn) + sizeof(float) - 1) / sizeof(float));
    float* moe_out = a.alloc<float>(moe_out_floats);
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
    // Width of the per-row MoE fan-out, counting the caller's stream. One row's MoE does not fill
    // the GPU, so issuing the rows on their own streams runs several at once. The rows are
    // independent (own input row, own expert slice, own scratch), so this changes only when the
    // launches run, never what they compute. Dedicated streams -- stream_k carries the shared
    // expert, which already overlaps the routed branch.
    static const int kRowFanout = []{
        const char* e = getenv("SPARKINFER_DFLASH_MOE_ROW_FANOUT");
        int v = e ? atoi(e) : 3;
        if (v < 1) v = 1;
        if (v > 4) v = 4;
        return v;
    }();
    static thread_local cudaStream_t row_stream[3] = {nullptr, nullptr, nullptr};
    static thread_local cudaEvent_t row_fork_ev[3] = {nullptr, nullptr, nullptr};
    static thread_local cudaEvent_t row_join_ev[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < kRowFanout - 1; ++i) if (!row_stream[i]) {
        pf_cu(cudaStreamCreateWithFlags(&row_stream[i], cudaStreamNonBlocking), "verify row stream");
        pf_cu(cudaEventCreateWithFlags(&row_fork_ev[i], cudaEventDisableTiming), "verify row fork ev");
        pf_cu(cudaEventCreateWithFlags(&row_join_ev[i], cudaEventDisableTiming), "verify row join ev");
    }
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
        // SI_QTYPE_FP8: the GDN projections have been checkpoint-native FP8 since #832 ("native
        // FP8 GDN GEMV"), which this verifier predates -- it only knew bf16/Q8_0/Q4_K/Q6_K, so a
        // Qwen3.8 verify died here after clearing the dense-FFN stage.
        //
        // Batched first, row loop as fallback. This was a bare row loop on the reasoning that AR
        // decode drives launch_gemv_fp8 at one row, so N one-row calls are bit-identical to N AR
        // steps by construction, and that any batched variant would reassociate the reduction and
        // put losslessness at risk "for a saving on the GDN projections, which are a small share of
        // the layer". The first half is sound; the second badly understated the cost. FP8 is not a
        // small share here -- on this target it is attention q/k/v/o and the linear-attn
        // projections across all 64 layers, the 248320x5120 lm_head, and layers 56-63's MLP -- and
        // since decode is memory-bound and W does not depend on the row, the loop re-read several
        // GB per extra row. That was the whole shape of the verify cost curve: 16.686 ms at N=1,
        // 16.740 at N=2, then 29.323 at N=4 and 53.465 at N=7 against an 11.162 ms single-token
        // forward, growing ~4 ms per row where a memory-bound decode should be nearly flat.
        // launch_gemv_fp8_rows keeps the reduction order rather than trading it away: same 8-wide
        // K-association, same per-j accumulation, same ordered split sum, and the same S-by-N
        // choice, so each row is bit-identical to the one-row call it replaces. It declines the
        // shapes it cannot serve that way, and those still take the loop below.
        if (type == kernels::SI_QTYPE_FP8) {
            if (kernels::launch_gemv_fp8_rows(in, w, out, N, no, k, st)) return true;
            for (int r = 0; r < N; ++r)
                kernels::launch_gemv_fp8(in + (size_t)r * k, w, out + (size_t)r * no, no, k, st);
            return true;
        }
        // SI_QTYPE_NVFP4: same story one checkpoint later. The ModelOpt export
        // (gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) quantizes ALL 400 Linears to NVFP4,
        // including every GDN projection, where the compressed-tensors export used FP8. Without
        // this branch that checkpoint declines the verify on layer 0 exactly as Qwen3.8 did
        // before the FP8 branch above was added.
        // Row loop for the same reason as FP8: AR decode drives launch_gemv_nvfp4 at one row, so
        // N one-row calls are bit-identical to N AR steps, which is what keeps the speculative
        // path lossless by construction rather than by measurement.
        if (type == kernels::SI_QTYPE_NVFP4) {
            // One weight stream for all N rows when a rows kernel covers this width. It reproduces
            // the one-row kernel's dot and split-fold order per row, so the results are the same
            // bits the loop below produces -- the loop stays as the fallback for widths it does
            // not cover, and is what every other N still runs.
            if (kernels::launch_gemv_nvfp4_rows(in, w, out, N, no, k, st)) return true;
            for (int r = 0; r < N; ++r)
                kernels::launch_gemv_nvfp4(in + (size_t)r * k, w, out + (size_t)r * no, no, k, st);
            return true;
        }
        if (type != 8 && type != 12 && type != 14) {
            fprintf(stderr, "[dflash-verify] unsupported projection type=%d N=%d K=%d\n", type, no, k);
            return false;
        }
        quant_rows(in, k);
        if (!kernels::launch_mmvq_rows(type, q81, w, out, N, no, k, st)) {
            // Distinguishes "this weight TYPE is not implemented" (handled above, prints its own
            // message) from "the mmvq launcher refused THIS SHAPE" -- which is otherwise a silent
            // false and reads identically at the call site.
            fprintf(stderr, "[dflash-verify] mmvq_rows refused type=%d N=%d n_out=%d K=%d\n",
                    type, N, no, k);
            return false;
        }
        return true;
    };
    // proj() on an arbitrary stream. Callers must have `in` already quantized into q81 (checked at
    // each call site), because quantizing here would write shared scratch off the main stream.
    auto proj_on = [&](cudaStream_t ps, const bf16* in, const void* w, int type, bf16* out,
                       int no, int k) -> bool {
        if (type == 0) return kernels::launch_gemv_rows(in, w, out, N, no, k, ps);
        // Native FP8/NVFP4 touch no shared q81 scratch, so unlike the mmvq path below they are
        // safe on ANY stream and never need the fall-back to `st`.
        if (type == kernels::SI_QTYPE_FP8 || type == kernels::SI_QTYPE_NVFP4) {
            if (type == kernels::SI_QTYPE_NVFP4 &&
                kernels::launch_gemv_nvfp4_rows(in, w, out, N, no, k, ps)) return true;
            if (type == kernels::SI_QTYPE_FP8 &&
                kernels::launch_gemv_fp8_rows(in, w, out, N, no, k, ps)) return true;
            for (int r = 0; r < N; ++r) {
                const bf16* xr = in + (size_t)r * k;
                bf16* yr = out + (size_t)r * no;
                if (type == kernels::SI_QTYPE_FP8) kernels::launch_gemv_fp8(xr, w, yr, no, k, ps);
                else                               kernels::launch_gemv_nvfp4(xr, w, yr, no, k, ps);
            }
            return true;
        }
        if (type != 8 && type != 12 && type != 14) return false;
        if (q81_src != in || q81_k != k) { quant_rows(in, k); ps = st; }
        return kernels::launch_mmvq_rows(type, q81, w, out, N, no, k, ps);
    };
    auto capture = [&](int layer) {
        if (!capture_dst || !capture_layers || n_capture <= 0) return;
        for (int slot = 0; slot < n_capture; ++slot) if (capture_layers[slot] == layer) {
            char* dst = static_cast<char*>(capture_dst) + (size_t)slot * H * sizeof(bf16);
            // Kernel node, not a 2-D memcpy node: same bytes to the same addresses, but it
            // schedules against its neighbours instead of draining the graph (see the note on
            // launch_capture_rows in dflash_kernels.h).
            dflash_kernels::launch_capture_rows(x, dst, N, H, n_capture * H, st);
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
    int  vfail_L = -1;   // layer whose stage declined, for the bailout diagnostic below
    bool recording = false;
    // Abandon an in-progress capture so the stream is usable again. EVERY early return between
    // "verify graph begin" and "verify graph end" must call this first: cudaStreamEndCapture is
    // the only way to leave cudaStreamCaptureStatusActive, so a bare return strands the stream in
    // capturing state permanently and every later CUDA call on it fails -- including the caller's
    // token-loop fallback, which is why a merely-unsupported shape presented as model corruption.
    // Declared HERE, above `goto verify_forward_done`'s target scope: a goto may not jump over a
    // variable initialization, and defining it later broke the build outright.
    // The partial graph is destroyed, never instantiated -- recorded mid-layer, not replayable.
    auto abandon_capture = [&]() {
        if (!recording) return;
        cudaGraph_t partial = nullptr;
        if (cudaStreamEndCapture(st, &partial) == cudaSuccess && partial) cudaGraphDestroy(partial);
        else cudaGetLastError();
        recording = false;
    };
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
        if (vdbg_dump_now) {
            std::vector<bf16> host((size_t)(c.n_layers + 1) * H);
            pf_cu(cudaMemcpy(host.data(), verify_dbg_buf, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
                  "verify_dbg readback (replay)");
            const char* path = getenv("SPARKINFER_DFLASH_VERIFY_DUMP_FILE");
            FILE* f = fopen(path ? path : "/tmp/dflash_verify_xn_dump.bin", "wb");
            if (f) { fwrite(host.data(), sizeof(bf16), host.size(), f); fclose(f); }
            fprintf(stderr, "[dflash-verify-debug] dumped xn[layer,H=%d] for row=%d start_pos=%d, %d layers -> %s (replay path)\n",
                    H, verify_dbg_row, start_pos, c.n_layers, path ? path : "/tmp/dflash_verify_xn_dump.bin");
            std::vector<bf16> host2((size_t)5 * H);
            pf_cu(cudaMemcpy(host2.data(), verify_dbg_buf2, host2.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
                  "verify_dbg2 readback (replay)");
            const char* path2 = getenv("SPARKINFER_DFLASH_VERIFY_DUMP_FILE2");
            FILE* f2 = fopen(path2 ? path2 : "/tmp/dflash_verify_l0_substages.bin", "wb");
            if (f2) { fwrite(host2.data(), sizeof(bf16), host2.size(), f2); fclose(f2); }
            fprintf(stderr, "[dflash-verify-debug] dumped layer0 substages [ao,h,hn,routed,x] -> %s (replay path)\n",
                    path2 ? path2 : "/tmp/dflash_verify_l0_substages.bin");
            verify_dbg_call++;
        }
        goto verify_forward_done;
    }
    recording = graph_warm || capture_only;
    if (recording)
    // Dense FFN seeds: expert 0, weight 1.0 -- the same constants AR uses. Written ONCE, here,
    // SYNCHRONOUSLY, and deliberately BEFORE the capture begins.
    //
    // This used to live inside the layer loop as a cudaMemcpyAsync from a `std::vector<float>`
    // declared in that block. Two independent defects: the vector destructs while an async copy
    // from pageable host memory may still be in flight, and -- fatally -- under stream capture the
    // copy is RECORDED with that host pointer, so every graph replay reads a long-dead stack
    // buffer. The expert weights came back as garbage and the FFN output as zeros. It never showed
    // until the mmvq K=5120 instantiation landed, because before that the verify declined at layer
    // 3 and this code had never actually run.
    //
    // Hoisting also makes it correct by construction rather than by care: these values are
    // constant across layers and across replays, so there is nothing for the graph to capture.
    if (dense && expert_ids && expert_w) {
        std::vector<float> ones((size_t)N * topk, 1.0f);
        pf_cu(cudaMemset(expert_ids, 0, (size_t)N * topk * sizeof(int)), "dense expert ids seed");
        pf_cu(cudaMemcpy(expert_w, ones.data(), ones.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "dense expert w seed");
    }
        pf_cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "verify graph begin");
    pf_cu(cudaMemcpyAsync(ids, ph_ids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "verify ids");
    pf_cu(cudaMemcpyAsync(pos, ph_pos, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "verify pos");
    pf_cu(cudaMemcpyAsync(seq, ph_seq, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "verify lens");
    // Device-to-device inside the capture, so each replay re-reads the sequence's live table as it
    // grows instead of baking in the mapping from capture time. One kernel node rather than N
    // memcpy nodes -- same reason as the capture copies above.
    if (btab_rows)
        dflash_kernels::launch_broadcast_rows_i32(btable, btab_rows, mbs, N, st);
    kernels::launch_embedding(ids, s.w.embed_tokens, x, N, H, st);
    kernels::launch_rmsnorm(x, s.w.layers[0].input_norm, xn, N, H, c.rms_eps, st);
    for (int L = 0; L < c.n_layers && supported; ++L) {
        vfail_L = L;
        vdbg_snapshot(xn, L);
        if (L == 0) vdbg_snapshot2(x, 4);   // raw pre-norm residual stream (h = x + ao)
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
                state, att, N, c.linear_q_heads, vh, c.linear_head_dim, c.gdn_qh_block, st);
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
                       s.kv->layer_base_elems(L) * kv_elem;
            char* vp = static_cast<char*>(s.kv->v_pool()) +
                       s.kv->layer_base_elems(L) * kv_elem;
            char* ks = kv8 ? static_cast<char*>(s.kv->k_scale_pool()) +
                             s.kv->scale_layer_base_elems(L) * 2 : nullptr;
            char* vs = kv8 ? static_cast<char*>(s.kv->v_scale_pool()) +
                             s.kv->scale_layer_base_elems(L) * 2 : nullptr;
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
            if (!kv8) {
                kernels::launch_qwen36_mul_sigmoid(att, qg, N * qdim, st);
            }
            supported = proj(att, w.wo, w.wo_type, ao, H, qdim);
        }
        if (!supported) break;
        // DEBUG ONLY: layer-0-only sub-stage capture (ao = GDN/attn output pre-residual, h =
        // post-residual, hn = pre-FFN norm) for the SAME row as verify_dbg_buf, into a second
        // small buffer -- memcpy destination pointers survive graph replay even though scalar
        // kernel args (a printf's tag/step) do not, which is why this uses vdbg_snapshot's
        // approach and not a live stat-printer call here.
        if (L == 0) vdbg_snapshot2(ao, 0);
        kernels::launch_add_rmsnorm2_q8_rows(x, ao, w.post_attn_norm, h, hn, q81,
                                             N, H, c.rms_eps, st);
        // FIXED (2026-08-17): this used to snapshot h BEFORE this kernel wrote it -- h is an
        // arena buffer reused every layer, so the earlier capture was reading whatever the
        // PREVIOUS layer's residual left there, not this layer's real value. That stale read is
        // what produced the ~456 magnitude "anomaly" that briefly looked like a kernel bug; an
        // isolated repro of this exact kernel under compute-sanitizer (memcheck + racecheck, 0
        // errors both) proved the kernel itself was never the problem.
        if (L == 0) { vdbg_snapshot2(h, 1); vdbg_snapshot2(hn, 2); }
        q81_src = hn; q81_k = H;

        // Dense FFN: one expert, no router, no shared expert. Seed ids/weights with the same
        // constants AR uses (expert 0, weight 1.0) and run the identical expert kernel at N rows,
        // then skip straight past the MoE routing/shared machinery below.
        if (dense) {
            // Same default and same env as the decode dispatch in qwen35.cpp -- ON unless
            // SPARKINFER_QWEN38_DECODE_NVFP4=0. These two MUST stay in lockstep; see below.
            static const bool kDecodeNvfp4 = [] {
                const char* e = getenv("SPARKINFER_QWEN38_DECODE_NVFP4");
                return !(e && e[0] == '0');
            }();
            const bool native_ffn = kDecodeNvfp4 && w.gate_nv && w.up_nv && w.down_nv;
            const bool q4_ffn = w.gate_q && w.up_q && w.down_q;
            if (!native_ffn && !q4_ffn) {
                fprintf(stderr, "[dflash-verify] dense layer=%d missing gate/up/down\n", L);
                supported = false; break;
            }
            // Checkpoint-native NVFP4 FFN, matching the decode path's own branch
            // (SPARKINFER_QWEN38_DECODE_NVFP4, see qwen35.cpp). BOTH sides have to switch
            // together: losslessness is defined as the speculative output matching the AR output
            // of the SAME build, so running NVFP4 in decode while the verify stays on Q4_K makes
            // the two disagree on weights that differ by ~8% and reports LOSSLESS=0 -- a path
            // inconsistency, not an accuracy finding. Measured exactly that way before this
            // branch existed.
            if (native_ffn) {
                const bool dual = topk == 1 && kernels::launch_gemv_nvfp4_rows_dual_swiglu(
                    hn, w.gate_nv, w.up_nv, sh, N, ffn, H, st);
                if (!dual && (topk != 1 ||
                    !kernels::launch_gemv_nvfp4_rows(hn, w.gate_nv, sg, N, ffn, H, st) ||
                    !kernels::launch_gemv_nvfp4_rows(hn, w.up_nv, su, N, ffn, H, st))) {
                    fprintf(stderr, "[dflash-verify] NVFP4 gate/up declined N=%d ffn=%d H=%d\n",
                            N, ffn, H);
                    supported = false; break;
                }
                if (!dual) kernels::launch_prefill_swiglu(sg, su, sh, (long)N * ffn, st);
                if (!kernels::launch_gemv_nvfp4_rows(sh, w.down_nv, routed, N, H, ffn, st)) {
                    fprintf(stderr, "[dflash-verify] NVFP4 down declined N=%d H=%d ffn=%d\n",
                            N, H, ffn);
                    supported = false; break;
                }
            } else {
                kernels::launch_moe_expert_ffn_q4k(hn, w.gate_q, w.up_q, w.down_q,
                                                   w.gate_qtype, w.up_qtype, w.down_qtype,
                                                   expert_ids, expert_w, routed, moe_h, moe_out,
                                                   N, topk, H, ffn, q81, st);
            }
            if (L == 0) vdbg_snapshot2(routed, 3);
            // Residual + next-layer norm, matching the MoE branch's tail.
            const void* nn = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
            kernels::launch_add_rmsnorm2_q8_rows(h, routed, nn, x, xn, q81,
                                                 N, H, c.rms_eps, st);
            q81_src = xn; q81_k = H;
            // capture(L) MUST still run: it is the last statement of the layer loop and it is what
            // hands this layer's hidden state to the draft. An earlier revision of this branch used
            // a bare `continue` and skipped it for every layer, so the draft was fed a stale capture
            // buffer -- proposals became garbage (tau pinned at 1.0) and the emitted stream
            // degenerated into one token repeated. Fall through instead of jumping.
            capture(L);
            continue;
        }

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
        // What made launch_moe_expert_ffn_q4k disagree with the num_tokens=1 call AR decode makes
        // was one row-count-dependent choice inside it: the Q5_K down projection picks its split
        // count from num_tokens (S=8 at one row, S=1 above), and S sets the reduction order. Same
        // input, same activations, same expert ids, same weights -- different rounding.
        //
        // Pinning S to AR's choice removes the dependence outright. With S fixed the down kernel
        // grids as dim3(num_tokens, ...), one block column per token, and the gate/up and quantize
        // kernels grid per (token, expert, column); every row then computes exactly what it would
        // as a one-row call, for any N. So the batched call is exact by construction too, and it
        // costs one launch instead of N.
        //
        // The per-row loop stays available as an escape hatch (SPARKINFER_DFLASH_MOE_ROWWISE=1),
        // since it is exact for reasons that do not depend on this reasoning being right.
        static const bool moe_rowwise = []{
            const char* e = getenv("SPARKINFER_DFLASH_MOE_ROWWISE");
            return e && e[0] == '1';
        }();
        // Scope: below kCompactMaxSeq this stays on the single batched call, which is what main
        // already ships and what #720 certified 48/48 exact there -- the per-row error is real but
        // never compounds far enough to flip an argmax over a short generation, and the row loop
        // would cost ~5% for nothing. Above the bound it does compound, so exactness is required.
        static const int kRowwiseMinSeq = []{
            const char* e = getenv("SPARKINFER_DFLASH_MOE_ROWWISE_MINSEQ");
            return e ? atoi(e) : 384;
        }();
        // Same bound decides the pinned split count, so short context keeps main's row-count aware
        // S=1 and stays byte-identical; only the long-context path, which needs to reproduce AR,
        // pays the pinned-S choice.
        const bool moe_exact_splitk = (start_pos + N) > kRowwiseMinSeq;
        if (moe_rowwise && (start_pos + N) > kRowwiseMinSeq) {
            // Give every row its own scratch slice. Sharing moe_h/moe_out across the loop makes the
            // calls false-dependent, so they serialize and the row loop costs ~28% at 4k; sliced,
            // they are genuinely independent and the captured graph can overlap them.
            const size_t q81_row = kernels::llama_q8_1_bytes(H);
            const size_t h_row   = (size_t)topk * ffn;
            const size_t out_row = moe_out_floats / (size_t)N;
            // The rows are independent (own input row, own expert slice, own scratch), but issued
            // back to back on one stream they serialize, and one row's MoE does not fill the GPU.
            // Fan them across a second stream so two rows are in flight at once. stream_v is idle
            // here -- the shared expert has stream_k -- and this changes only when the launches
            // run, never what they compute.
            const int fan = std::min(kRowFanout, N);
            for (int i = 0; i < fan - 1; ++i) {
                pf_cu(cudaEventRecord(row_fork_ev[i], st), "verify moe row fork");
                pf_cu(cudaStreamWaitEvent(row_stream[i], row_fork_ev[i], 0), "verify moe row fork wait");
            }
            for (int r = 0; r < N; ++r) {
                const int slot = r % fan;
                cudaStream_t rs = slot == 0 ? st : row_stream[slot - 1];
                kernels::launch_moe_expert_ffn_q4k(hn + (size_t)r * H, w.gate_q, w.up_q, w.down_q,
                    w.gate_qtype, w.up_qtype, w.down_qtype,
                    expert_ids + (size_t)r * topk, expert_w + (size_t)r * topk,
                    routed + (size_t)r * H, moe_h + (size_t)r * h_row,
                    moe_out + (size_t)r * out_row, 1, topk, H, ffn,
                    static_cast<const unsigned char*>(q81) + (size_t)r * q81_row, rs);
            }
            for (int i = 0; i < fan - 1; ++i) {
                pf_cu(cudaEventRecord(row_join_ev[i], row_stream[i]), "verify moe row join");
                pf_cu(cudaStreamWaitEvent(st, row_join_ev[i], 0), "verify moe row join wait");
            }
        } else
        kernels::launch_moe_expert_ffn_q4k(hn, w.gate_q, w.up_q, w.down_q,
            w.gate_qtype, w.up_qtype, w.down_qtype, expert_ids, expert_w, routed,
            moe_h, moe_out, N, topk, H, ffn, q81, st, moe_exact_splitk);

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
    if (!supported) {
        // Name the layer. Every decline that has an obvious cause already prints one (missing
        // dense weights, unsupported MoE shape, unsupported projection TYPE); reaching here with
        // none of those means a kernel launcher itself returned false, which is otherwise silent
        // and indistinguishable from "this path is simply not implemented for your model".
        fprintf(stderr, "[dflash-verify] declined at layer=%d (linear_attn=%d) N=%d start=%d\n",
                vfail_L, (vfail_L >= 0 && vfail_L < (int)s.w.layers.size())
                             ? (int)s.w.layers[vfail_L].linear_attn : -1,
                N, start_pos);
        // Reached when a layer's projection type is one proj() cannot drive (anything but
        // 0/8/12/14 -- notably SI_QTYPE_FP8 108 and SI_QTYPE_NVFP4 109, i.e. EVERY GDN layer of a
        // compressed-tensors Qwen3.8 checkpoint). Falling back to the token loop is correct; doing
        // so with the capture still open is what turned a missing optimisation into garbage output.
        abandon_capture();
        return -1;
    }

    vdbg_snapshot(xn, c.n_layers);   // extra slot: post-final-norm xn, matching qwen35.cpp's convention
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
    if (verify_head_i8_on && verify_head_i8) {
        head_ok = kernels::launch_gemv_i8_q81_multirow_f32(
            q81, verify_head_i8, verify_head_scale, logits, c.vocab, H, N, st);
    } else if (s.w.lm_head_type == 12) {
        // AR decode drives the LM head with launch_mmvq_q4k_f32 at one row (qwen35.cpp:1888);
        // this makes the verify call the identical kernel instead of launch_mmvq_rows_f32's
        // batched Q4_K path, on the same "N one-row AR calls are bit-identical by construction"
        // reasoning as the FP8/NVFP4 GDN projections above.
        //
        // NOTE (2026-08-17): this alone does NOT make dspark_tau_check report lossless=YES.
        // si_mmvq_q4k_rows_exact_kernel's own per-row dot/reduction order claim held up under
        // direct testing -- swapping every Q4_K GEMV in the model (not just this head) to the
        // same per-row AR kernel, AND separately pinning SPARKINFER_NSPLITS=1 to remove
        // flash-decode's split-K nondeterminism (see dspark_tau_check.cpp), left the FIRST
        // divergence (position 3, verify=3065 vs AR=314, K=5120/Qwen3.8-27B) byte-for-byte
        // unchanged across all three variants. That rules out Q4_K kernel choice and split-K
        // order as the cause of THIS divergence -- something else (attention/GDN state handling
        // most likely, see the #712 precedent noted at the flash_decode_split call below) is the
        // real source. Kept anyway: it is still a strictly more defensible position than trusting
        // an unverified "same order" claim, and costs nothing extra since N is small here.
        // UPDATE (2026-08-20): the "costs nothing extra since N is small here" above is wrong, and
        // it was the single largest remaining per-row cost in the verify. This head is
        // 248320x5120 Q4_K, ~0.72 GB, and nsys clocks one row at 423 us -- 1.70 TB/s, squarely
        // DRAM-bound -- so the loop re-read the whole head once per candidate row: 1.7 ms of a
        // 21.6 ms verify at N=4, and 2.9 ms of 35.2 ms at N=7.
        //
        // launch_mmvq_rows_f32 covers exactly this case (qtype 12, K=5120) through
        // si_mmvq_q4k_rows_exact_kernel -- the same kernel whose per-row dot/reduction order the
        // note above already tested and found to leave the then-current divergence byte-for-byte
        // unchanged, i.e. it reproduces the per-row AR order. It reads the head once for all N
        // rows. The loop stays as the fallback for any shape it declines, and
        // SPARKINFER_DFLASH_VERIFY_HEAD_ROWS=0 restores it outright.
        static const bool head_rows_on = [] {
            const char* e = getenv("SPARKINFER_DFLASH_VERIFY_HEAD_ROWS");
            return !(e && e[0] == '0');
        }();
        head_ok = head_rows_on && N > 1 &&
                  kernels::launch_mmvq_rows_f32(s.w.lm_head_type, q81, s.w.lm_head, logits,
                                                N, c.vocab, H, st);
        if (!head_ok) {
            const size_t q81_row_bytes = kernels::llama_q8_1_bytes(H);
            for (int r = 0; r < N; ++r) {
                kernels::launch_mmvq_q4k_f32(
                    static_cast<const unsigned char*>(q81) + (size_t)r * q81_row_bytes,
                    s.w.lm_head, logits + (size_t)r * c.vocab, c.vocab, H, st);
            }
            head_ok = true;
        }
    } else {
        head_ok = kernels::launch_mmvq_rows_f32(
            s.w.lm_head_type, q81, s.w.lm_head, logits, N, c.vocab, H, st);
    }
    if (!head_ok) {
        fprintf(stderr, "[dflash-verify] unsupported LM head type=%d H=%d\n", s.w.lm_head_type, H);
        abandon_capture();
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
    if (vdbg_dump_now) {
        std::vector<bf16> host((size_t)(c.n_layers + 1) * H);
        pf_cu(cudaMemcpy(host.data(), verify_dbg_buf, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
              "verify_dbg readback");
        const char* path = getenv("SPARKINFER_DFLASH_VERIFY_DUMP_FILE");
        FILE* f = fopen(path ? path : "/tmp/dflash_verify_xn_dump.bin", "wb");
        if (f) { fwrite(host.data(), sizeof(bf16), host.size(), f); fclose(f); }
        fprintf(stderr, "[dflash-verify-debug] dumped xn[layer,H=%d] for row=%d start_pos=%d, %d layers -> %s\n",
                H, verify_dbg_row, start_pos, c.n_layers, path ? path : "/tmp/dflash_verify_xn_dump.bin");
        std::vector<bf16> host2((size_t)5 * H);
        pf_cu(cudaMemcpy(host2.data(), verify_dbg_buf2, host2.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
              "verify_dbg2 readback");
        const char* path2 = getenv("SPARKINFER_DFLASH_VERIFY_DUMP_FILE2");
        FILE* f2 = fopen(path2 ? path2 : "/tmp/dflash_verify_l0_substages.bin", "wb");
        if (f2) { fwrite(host2.data(), sizeof(bf16), host2.size(), f2); fclose(f2); }
        fprintf(stderr, "[dflash-verify-debug] dumped layer0 substages [ao,h,hn,routed,x] -> %s\n",
                path2 ? path2 : "/tmp/dflash_verify_l0_substages.bin");
        verify_dbg_call++;
    }
verify_forward_done:
    int keep = 1;
    while (keep < N && token_ids[keep] == out_argmax[keep - 1]) ++keep;
    if (getenv("SPARKINFER_DFLASH_VERIFY_DUMP_ROW")) {
        fprintf(stderr, "[dflash-verify-debug] start_pos=%d N=%d keep=%d out_argmax=[", start_pos, N, keep);
        for (int i = 0; i < N; i++) fprintf(stderr, "%d ", out_argmax[i]);
        fprintf(stderr, "]\n");
    }
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
            n_gdn, keep, c.linear_q_heads, vh, c.linear_head_dim, c.gdn_qh_block, st);
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
                s.w.layers[L].ssm_a, state, keep, c.linear_q_heads, vh, c.linear_head_dim,
                c.gdn_qh_block, st);
        }
    }
    pf_cu(cudaStreamSynchronize(st), "verify commit");
    return keep;
}

} // namespace sparkinfer
