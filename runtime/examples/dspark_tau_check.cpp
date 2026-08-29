// What acceptance length (tau) does the DSpark draft achieve against Qwen3.8-27B?
//
// Originally the number that decided whether the remaining speculative-decoding work was worth
// doing. DSpark's block_size is 7, so tau can in principle reach 7 -- against MTP's hard ceiling
// of 2 (mtp_num_hidden_layers=1). The framing then was "tau ~4-5 and DSpark clearly wins; ~2 and
// MTP is simpler for the same benefit". Tau landed well below that band (~1.66 at ctx=4k) and
// DSpark still won, because the payoff turned out to hinge on VERIFY COST rather than tau alone:
// speedup ~= tau / (verify + draft), and #889 cut verify from 2.62 to 1.85 target forwards at
// N=4. Do not read the old tau bands as a live decision rule.
//
// Tau was measured BEFORE the expensive part was built, deliberately: both speculative paths were
// gated behind dflash_verify_short_run rejecting dense_ffn (its guards wanted !c.dense_ffn and
// n_experts==256, written for Qwen3.6-35B-A3B's MoE), and Qwen3.8 is dense_ffn with n_experts==1.
// That branch now exists, so this reports THROUGHPUT as well as tau and losslessness, and is the
// harness the hourly DSpark eval scores -- at ctx=4096 (dspark-decode@4k), not 128.
//
// Read the three together. Throughput conditional on lossless=YES is the only meaningful number --
// a speculative decoder that emits unverified tokens is not fast, it is wrong.
//
// This header used to end "which is why DSPARK_TPS currently lands BELOW AR_TPS". That has not
// been true since the verify-cost work: at ctx=4096 DSpark measures ~1.27x AR, losslessly. Tau is
// still the lever, but it is the numerator of tau/(verify+draft), and the denominator moved
// further. Expect DSPARK_TPS above AR_TPS at 4k; below it at short context, where the batched
// verify cannot arm and the draft is pure overhead.
//
// The env pins below are deliberately NOT production defaults; they exist to make AR and DSpark
// comparable in one process. Both legs run under the same pins and the scored metric is relative
// (PR vs main on the same box, same pins), so the offset cancels -- but do not read AR_TPS here as
// the serving decode rate, which qwen3_gguf_bench measures without them.
//
// Usage: dspark_tau_check <qwen38_checkpoint_dir> <dspark_draft_dir> <max_new> <id0> [id1 ...]

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <qwen38_dir> <dspark_dir> <max_new> <id0> [id1 ...]\n", argv[0]);
        return 2;
    }
    // NO SPARKINFER_NSPLITS PIN. There used to be one (=1), and removing it on 2026-08-19 is the
    // single most consequential change this harness has had.
    //
    // Its justification was that "flash-decode's split-K reduction is atomic and therefore
    // order-dependent", citing 27-32/32 agreement at adaptive splits versus 32/32 pinned. That was
    // wrong, and the error was one of attribution. Read fa_combine_kernel: each warp folds a fixed
    // strided subset of splits in ascending order, then warps are folded in ascending index, and
    // there is not a single atomic in flash_decode_split.cu. The decode combine is a fixed-order
    // reduction for a given n_splits. The nondeterminism that experiment measured was real but came
    // from the TWO ATOMIC SPLIT-K PREFILL GEMMS pinned immediately below -- which were still live
    // when it was run, and were only root-caused afterwards. Decode was blamed for prefill's flake.
    //
    // The cost of that mistake was not academic. Pinning n_splits to 1 leaves the flash-decode grid
    // a fraction of the machine wide, and measured on this box at ctx=4096 it costs 35% of decode
    // throughput: 93.40 tok/s at the split count the engine selects, 60.43 pinned. The eval scored
    // every PR in that regime -- a configuration nothing serves -- and in one day merged three
    // attention PRs that measured +11.8%, +18.6% and +11.7% against it (#872, #874, #877) while
    // production decode at 4k went 93.69 -> 93.64. Three tiers awarded, nothing delivered. Two
    // separate contributors (#871, #875) diagnosed the pin correctly before we did.
    //
    // So: splits are left adaptive, exactly as the server runs them, and throughput measured here
    // is throughput that transfers. An explicit SPARKINFER_NSPLITS in the environment still wins
    // for anyone deliberately sweeping, and setting it also disables adaptive selection -- so do
    // not set it in the eval path.
    //
    // Determinism is verified rather than assumed: see the AR-REPS and SPEC-REPS probes below, and
    // the losslessness verdict now requires every repeat to match, not just the first.
    // Same story, second and third instances (2026-08-17): TWO more atomic split-K prefill GEMMs,
    // found only after a multi-round dflash_generate run that looked individually correct at
    // every kernel level (LM head, GDN conv/scan/commit all independently verified bit-exact
    // against AR) still diverged nondeterministically -- different divergence points across
    // identical repeat runs of the SAME prompt, proof it was hardware split-K noise and not a
    // logic bug.
    //   - prefill_gemm_i8.cu: atomicAdd into P[M,N]. int8 prefill is ON by default for dense
    //     models (use_i8 = !moe), so this needs an explicit override.
    setenv("SPARKINFER_PREFILL_I8", "0", 0);
    //   - prefill_gemm_skinny.cu: a SEPARATE atomicAdd split-K path that launch_prefill_gemm
    //     tries FIRST for narrow-N GEMMs (kernels/csrc/cuda/fused/batched_prefill.cu:1191) --
    //     runs unconditionally for the bf16 fallback too, so disabling int8 prefill alone was not
    //     enough (still 1/3 runs diverged). This is exactly Qwen3.8-27B's shape: its 48-v-head
    //     ssm_alpha/ssm_beta GDN gate projections are what the kernel's own comments call out as
    //     the motivating case. SPLITK=0 keeps the optimized narrow-N kernel, only turns off its
    //     atomic accumulation. 5/5 lossless=YES on both the 32- and 40-token repros after both
    //     pins landed (was still nondeterministic with only the first).
    setenv("SPARKINFER_PREFILL_SKINNY_SPLITK", "0", 0);
    // Batched long-context prefill needs the checkpoint-native NVFP4 FFN operands. At the scored
    // 16k context this fits comfortably on a 5090. Pin it for this evaluator so a caller's
    // low-memory serving environment cannot select the unsupported Q4_K batched-FFN fallback.
    setenv("SPARKINFER_QWEN38_PREFILL_NVFP4", "1", 1);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string tpath = argv[1], dpath = argv[2];
    const int max_new = atoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));

    sparkinfer::GGUF g;
    sparkinfer::Qwen35Config cfg;
    QwenCheckpointKind kind{};
    std::string err;
    if (!qwen_checkpoint_open(tpath, cfg, g, kind, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    const char* min_maxseq_env = getenv("SPARKINFER_DSPARK_MIN_MAXSEQ");
    cfg.max_seq = std::max(min_maxseq_env ? atoi(min_maxseq_env) : 2048,
                           (int)prompt.size() + max_new + 64);

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16; kvc.int8_kv = false;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    auto memck = [](const char* label) {
        if (!getenv("SPARKINFER_DSPARK_MEMCK")) return;
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        fprintf(stderr, "[memck] %-24s free=%.3fGB total=%.3fGB used=%.3fGB\n",
                label, f / 1e9, t / 1e9, (t - f) / 1e9);
    };

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    memck("after KV alloc");
    printf("loading target (%s) ...\n", qwen_checkpoint_kind_label(kind));
    if (!qwen_checkpoint_load(model, tpath, kind)) { printf("[FAIL] target load\n"); return 1; }
    memck("after target load");

    sparkinfer::DFlashDraftConfig dcfg;
    // The draft must cover the same per-run token budget as the target. Its legacy 8192 default
    // otherwise fails at an exactly-8k prompt before the first proposal (the B-token block needs
    // headroom beyond the prompt), even though the target was sized correctly above.
    dcfg.max_seq = cfg.max_seq;
    sparkinfer::DFlashDraftModel draft(dcfg);
    if (!draft.load(dpath)) { printf("[FAIL] draft load\n"); return 1; }
    memck("after draft load");
    const sparkinfer::DFlashDraftConfig& dc = draft.config();

    // The draft has no embed_tokens / lm_head of its own -- it borrows the target's, which is what
    // makes its proposals directly comparable to what the target would produce.
    draft.set_shared_weights(model.embed_weights(), model.lm_head_weights(),
                             model.lm_head_quant_type(), cfg.vocab, cfg.hidden);
    memck("after set_shared_weights");
    draft.ensure_quant();
    memck("after ensure_quant");

    // Capture layers come from the CHECKPOINT, not from this runtime's Qwen3.6 default: DSpark
    // projects [4,16,28,40,52] and its fc is sized n_cap*hidden accordingly.
    model.set_dflash_draft(&draft);
    // max_rows: use the API default (16), NOT block_size. dflash_generate writes capture rows
    // 0..kProposalDepth, and kProposalDepth is chosen from the SEQUENCE LENGTH (5 short / 7 long),
    // not from the draft's block_size -- so sizing this buffer from block_size is sizing it from
    // the wrong quantity entirely, and it can be overrun.
    model.set_dflash_capture(true, dc.target_layer_ids);
    memck("after set_dflash_capture");
    printf("draft: layers=%d B=%d n_cap=%zu  target: layers=%d dense_ffn=%d experts=%d\n",
           dc.n_layers, dc.block_size, dc.target_layer_ids.size(),
           cfg.n_layers, (int)cfg.dense_ffn, cfg.n_experts);

    // AR reference FIRST, on clean state. Taking it afterwards produced an EMPTY reference:
    // dflash_generate opens its own session via open_session(), so a following generate() starts
    // against a pool that still holds it, and freeing session 0 (which dflash never used) does not
    // release it. Ordering the runs this way avoids depending on teardown semantics the harness
    // does not control.
    model.set_dflash_draft(nullptr);
    model.set_dflash_capture(false, {}, 0);
    // Time the reference too. DECODE-only (not wall clock): the pins above force AR's prefill onto
    // the token loop for comparability, so a wall-clock rate here would report a deliberately
    // pessimised prefill, not decode. Both this and DSpark's stats.decode_s below exclude prefill,
    // which is what makes AR_TPS and DSPARK_TPS directly comparable -- and what lets this harness
    // report throughput at all, which it previously declined to do because the dense-FFN verify
    // branch did not exist yet (see the header comment).
    double ar_ttft_s = 0, ar_decode_s = 0;
    const std::vector<int> ar = model.generate(prompt, max_new, nullptr, &ar_ttft_s, &ar_decode_s);
    if (ar.empty()) { printf("[FAIL] AR reference produced nothing\n"); return 1; }
    const double ar_tps = ar_decode_s > 0 ? (double)ar.size() / ar_decode_s : 0.0;

    // AR determinism probe (SPARKINFER_DSPARK_AR_REPS=N). Repeats the SAME reference generation in
    // this one already-loaded process and reports any rep that disagrees with the first -- the
    // cheap way to chase a rare flake, since a fresh process per attempt spends ~25s reloading 22GB
    // of weights to get one sample. Exists because a prose run reported lossless=NO while tau was
    // bit-identical to its clean repeats: tau is a property of the speculative side alone, so an
    // unchanged tau alongside a changed verdict points at the REFERENCE having moved, not the
    // draft. This probe is what tells those two apart instead of inferring it.
    //
    // Each rep clears the prefix cache first, so every rep runs the identical no-reuse path that
    // produced `ar` above; SPARKINFER_DSPARK_AR_REPS_KEEP_PREFIX=1 leaves it warm instead, which
    // deliberately lets generate()'s prompt_matches_prefix() reuse branch run -- a different code
    // path, and worth testing separately rather than conflating with raw kernel determinism.
    if (const char* reps_env = getenv("SPARKINFER_DSPARK_AR_REPS")) {
        const int reps = atoi(reps_env);
        const bool keep_prefix = getenv("SPARKINFER_DSPARK_AR_REPS_KEEP_PREFIX") != nullptr;
        int mismatches = 0;
        for (int r = 1; r < reps; r++) {
            if (!keep_prefix) model.clear_prefix_cache();
            const std::vector<int> again = model.generate(prompt, max_new);
            if (again.empty()) { printf("AR-REPS rep %d produced nothing\n", r); mismatches++; continue; }
            size_t nn = std::min(ar.size(), again.size()), same = 0;
            while (same < nn && ar[same] == again[same]) same++;
            if (same != nn || ar.size() != again.size()) {
                mismatches++;
                printf("AR-REPS rep %d DIVERGED at %zu/%zu (len %zu vs %zu)\n",
                       r, same, nn, ar.size(), again.size());
                size_t lo = same >= 3 ? same - 3 : 0, hi = std::min(nn, same + 5);
                printf("  ref [");
                for (size_t i = lo; i < hi; i++) printf(" %d", ar[i]);
                printf(" ]\n  rep [");
                for (size_t i = lo; i < hi; i++) printf(" %d", again[i]);
                printf(" ]\n");
            }
        }
        printf("AR-REPS %d reps, %d mismatches, keep_prefix=%d\n", reps, mismatches, (int)keep_prefix);
    }

    // Prefix-reuse GDN probe (SPARKINFER_DSPARK_PREFIX_PROBE=1). cache_prefix() is the server's
    // warm-system-prompt path (server/src/model_engine.cpp:499), and for a hybrid model its own
    // comment claims it "retains KV + GDN state". That holds at cache time -- the GDN recurrent
    // state right after ingesting the prefix IS the state for that prefix. But a reuse generate()
    // then decodes onward and ADVANCES that same state past the prefix, and nothing restores it,
    // so the NEXT reuse re-serves prefix KV against a recurrent state belonging to a longer
    // sequence. KV tolerates this (append-only per position, stale tail never read below seqlen);
    // GDN cannot -- it is one running summary, not per-position.
    //
    // Both checks below must hold for a correct implementation:
    //   g1 == ar : caching a prefix must not change what the model generates at all
    //   g2 == ar : and it must still hold on the second reuse of the same prefix
    // A g1==ar / g2!=ar split is the advance-without-restore bug specifically.
    if (getenv("SPARKINFER_DSPARK_PREFIX_PROBE") && prompt.size() >= 4) {
        model.set_dflash_draft(nullptr);
        model.set_dflash_capture(false, {}, 0);
        const std::vector<int> pfx(prompt.begin(), prompt.begin() + prompt.size() / 2);
        if (!model.cache_prefix(pfx)) {
            printf("PREFIX-PROBE cache_prefix(%zu) failed\n", pfx.size());
        } else {
            printf("PREFIX-PROBE cached %zu of %zu prompt tokens, reuse_matches=%d\n",
                   pfx.size(), prompt.size(), (int)model.prompt_matches_prefix(prompt));
            auto state = [&](const char* when) {
                printf("  [state %s] reuse_matches=%d active_session=%llu\n", when,
                       (int)model.prompt_matches_prefix(prompt),
                       (unsigned long long)model.active_session());
            };
            auto cmp = [&](const char* tag, const std::vector<int>& got) {
                size_t nn = std::min(ar.size(), got.size()), same = 0;
                while (same < nn && ar[same] == got[same]) same++;
                const bool ok = (same == nn && ar.size() == got.size());
                printf("PREFIX-PROBE %s vs plain AR: %s (%zu/%zu)\n", tag,
                       ok ? "MATCH" : "DIVERGED", same, nn);
                if (!ok) {
                    size_t lo = same >= 3 ? same - 3 : 0, hi = std::min(nn, same + 5);
                    printf("  ar  [");
                    for (size_t i = lo; i < hi; i++) printf(" %d", ar[i]);
                    printf(" ]\n  %-4s[", tag);
                    for (size_t i = lo; i < hi; i++) printf(" %d", got[i]);
                    printf(" ]\n");
                }
            };
            // Three reuses, not two: the original bug showed up only on the SECOND, so a fix that
            // merely restores once needs a third to prove the state does not drift again.
            state("before g1"); cmp("g1", model.generate(prompt, max_new));
            state("before g2"); cmp("g2", model.generate(prompt, max_new));
            state("before g3"); cmp("g3", model.generate(prompt, max_new));
        }
        model.clear_prefix_cache();
    }

    // Isolation: does hidden-state CAPTURE alone perturb the target's decode? The speculative
    // loop differs from plain AR in exactly two ways -- capture is on, and a draft proposes. If AR
    // with capture enabled already diverges from AR without it, the fault is in the capture path
    // and nothing about the draft or the verify is implicated.
    model.set_dflash_capture(true, dc.target_layer_ids);
    const std::vector<int> ar_cap = model.generate(prompt, max_new);
    size_t capn = std::min(ar.size(), ar_cap.size()), capsame = 0;
    while (capsame < capn && ar[capsame] == ar_cap[capsame]) capsame++;
    printf("CAPTURE-only AR: matched %zu/%zu vs plain AR  [", capsame, capn);
    for (size_t i = 0; i < std::min<size_t>(8, ar_cap.size()); i++) printf(" %d", ar_cap[i]);
    printf(" ]\n");

    // Capture-ON AR determinism probe (SPARKINFER_DSPARK_ARCAP_REPS=N). AR-REPS above runs with
    // capture OFF, so it exercises the plain decode graph -- but the token-loop VERIFY drives
    // forward_token with dflash capture ON, which is a different graph plus the extra
    // launch_capture_rows write into dflash_hidden. The compact verify never uses that path at all
    // (it captures inside its own batched graph instead), which is exactly the asymmetry between
    // the two verify modes. This probe repeats the capture-ON generation with NO draft involved, so
    // a mismatch here localises the nondeterminism to the target's own capture-decode path rather
    // than to the draft or to the overlap between them.
    if (const char* creps_env = getenv("SPARKINFER_DSPARK_ARCAP_REPS")) {
        const int creps = atoi(creps_env);
        int cmis = 0;
        for (int r = 1; r < creps; r++) {
            model.clear_prefix_cache();
            const std::vector<int> again = model.generate(prompt, max_new);
            if (again.empty()) { printf("ARCAP-REPS rep %d produced nothing\n", r); cmis++; continue; }
            size_t nn = std::min(ar_cap.size(), again.size()), sm = 0;
            while (sm < nn && ar_cap[sm] == again[sm]) sm++;
            if (sm != nn || ar_cap.size() != again.size()) {
                cmis++;
                printf("ARCAP-REPS rep %d DIVERGED at %zu/%zu\n", r, sm, nn);
            }
        }
        printf("ARCAP-REPS %d reps, %d mismatches (capture ON, no draft)\n", creps, cmis);
    }

    model.set_dflash_draft(&draft);
    draft.reset();
    sparkinfer::Qwen35Model::DFlashStats stats;
    const std::vector<int> spec = model.dflash_generate(prompt, max_new, &stats);
    if (spec.empty()) { printf("[FAIL] dflash_generate produced nothing\n"); return 1; }

    // Speculative determinism probe (SPARKINFER_DSPARK_SPEC_REPS=N), the mirror of AR-REPS above.
    // The lossless verdict compares AR against SPEC, so a flake in EITHER produces the same
    // "lossless=NO" line -- and AR-REPS already showed the reference side reproducing exactly
    // (30/30 in-process, 20/20 across fresh processes), which leaves this side as the untested
    // half. Each rep resets the draft and re-runs the whole speculative generation, so a rep that
    // disagrees with the first localises a rare divergence to the draft/verify path rather than to
    // the reference.
    // Hoisted out of the probe block below so the VERDICT can see them. Before 2026-08-18 this
    // probe only printed: METRIC LOSSLESS compared AR against the FIRST speculative run and
    // nothing else, so a divergence that only shows up on some runs was reported in the log and
    // then scored as a pass. That is the exact shape of the bug this harness exists to catch --
    // the target/draft overlap defect diverged on 5-7 of 40 repeats, so a single-shot check finds
    // it about 15% of the time. Measured directly: with OVERLAP=1 deliberately in an auto-tuner's
    // search space, 10 of 11 trials that enabled it were scored LOSSLESS.
    int spec_rep_failures = 0;
    int spec_reps_run = 0;
    if (const char* sreps_env = getenv("SPARKINFER_DSPARK_SPEC_REPS")) {
        const int sreps = atoi(sreps_env);
        int smismatch = 0, vs_ar_mismatch = 0;
        for (int r = 1; r < sreps; r++) {
            draft.reset();
            sparkinfer::Qwen35Model::DFlashStats st2;
            const std::vector<int> again = model.dflash_generate(prompt, max_new, &st2);
            if (again.empty()) { printf("SPEC-REPS rep %d produced nothing\n", r); smismatch++; continue; }
            size_t nn = std::min(spec.size(), again.size()), sm = 0;
            while (sm < nn && spec[sm] == again[sm]) sm++;
            if (sm != nn || spec.size() != again.size()) {
                smismatch++;
                printf("SPEC-REPS rep %d DIFFERS from first spec at %zu/%zu (tau %.3f vs %.3f)\n",
                       r, sm, nn, stats.mean_accept, st2.mean_accept);
                size_t lo = sm >= 3 ? sm - 3 : 0, hi = std::min(nn, sm + 5);
                printf("  spec0 [");
                for (size_t i = lo; i < hi; i++) printf(" %d", spec[i]);
                printf(" ]\n  rep   [");
                for (size_t i = lo; i < hi; i++) printf(" %d", again[i]);
                printf(" ]\n");
            }
            // Independently of matching the first spec run, every rep must still match AR --
            // that is the actual lossless guarantee, and a rep could in principle drift from
            // spec0 while both remain valid only if AR itself moved (which AR-REPS rules out).
            size_t an = std::min(ar.size(), again.size()), am = 0;
            while (am < an && ar[am] == again[am]) am++;
            if (am != an) {
                vs_ar_mismatch++;
                printf("SPEC-REPS rep %d NOT LOSSLESS vs AR at %zu/%zu\n", r, am, an);
            }
        }
        printf("SPEC-REPS %d reps, %d differ-from-first, %d not-lossless-vs-AR\n",
               sreps, smismatch, vs_ar_mismatch);
        // Both count. differ-from-first catches a rep that drifted or produced nothing;
        // not-lossless-vs-AR is the guarantee itself. Either is a failure of the run as a whole.
        spec_reps_run = sreps > 1 ? sreps : 0;
        spec_rep_failures = smismatch + vs_ar_mismatch;
    }

    // An empty AR reference must FAIL, not pass vacuously: min(0, k) == 0 makes "matched 0/0"
    // satisfy same==n and report lossless=YES while comparing nothing at all. Observed for real --
    // generate() returned no tokens and the run still claimed success.
    if (ar.empty() || spec.empty() || ar.size() < spec.size()) {
        printf("DSPARK lossless=UNKNOWN  ar=%zu spec=%zu -- reference unusable, not a pass\n",
               ar.size(), spec.size());
        return 1;
    }
    size_t n = std::min(ar.size(), spec.size()), same = 0;
    while (same < n && ar[same] == spec[same]) same++;
    // A speculative path is lossless BY CONSTRUCTION -- every emitted token is a target argmax --
    // so a mismatch at index 0 does not mean "poor acceptance", it means one of these two runs is
    // not doing what it claims. Print both prefixes so the failing side is identifiable instead of
    // inferred: if AR here disagrees with a fresh AR run, the reference teardown is at fault; if
    // the speculative side is the odd one out, tokens are being committed unverified.
    printf("  AR  [");
    for (size_t i = 0; i < std::min<size_t>(8, ar.size()); i++) printf(" %d", ar[i]);
    printf(" ]\n  SPEC[");
    for (size_t i = 0; i < std::min<size_t>(8, spec.size()); i++) printf(" %d", spec[i]);
    printf(" ]\n");
    // Full streams for cross-runtime differential debugging. Keep this opt-in: routine eval logs
    // only need the compact prefix above, while a parity investigation needs every token id.
    if (getenv("SPARKINFER_DSPARK_DUMP_TOKENS")) {
        printf("DSPARK_AR_TOKENS");
        for (int token : ar) printf(" %d", token);
        printf("\nDSPARK_SPEC_TOKENS");
        for (int token : spec) printf(" %d", token);
        printf("\n");
    }
    if (same < n) {
        size_t lo = same >= 3 ? same - 3 : 0, hi = std::min(n, same + 5);
        printf("  first divergence at index %zu, window [%zu,%zu):\n  AR  [", same, lo, hi);
        for (size_t i = lo; i < hi; i++) printf(" %d", ar[i]);
        printf(" ]\n  SPEC[");
        for (size_t i = lo; i < hi; i++) printf(" %d", spec[i]);
        printf(" ]\n");
    }
    printf("DSPARK tau=%.3f  steps=%d  tokens=%zu  decode_s=%.3f\n",
           stats.mean_accept, stats.steps, spec.size(), stats.decode_s);
    // The verdict is ALL reps, not just the first. Losslessness is a property of the path, not of
    // one lucky run, and the defects this catches are probabilistic by nature: a race between the
    // draft and verify row 0 diverged on 5-7 of 40 repeats, which a single run misses ~85% of the
    // time. With SPEC_REPS unset this is exactly the old behaviour (spec_rep_failures stays 0), so
    // an unset harness is no weaker than before -- it is simply not any stronger.
    const bool lossless = (same == n) && spec_rep_failures == 0;
    printf("DSPARK lossless=%s  matched %zu/%zu", lossless ? "YES" : "NO", same, n);
    if (spec_reps_run > 0)
        printf("  (+%d repeats, %d failed)", spec_reps_run - 1, spec_rep_failures);
    else
        printf("  (single run -- set SPARKINFER_DSPARK_SPEC_REPS=N to test reproducibility)");
    printf("\n");
    printf("DSPARK ceiling: block_size=%d (MTP's ceiling is 2)\n", dc.block_size);

    const double spec_tps = stats.decode_s > 0 ? (double)spec.size() / stats.decode_s : 0.0;
    printf("AR     %.2f tok/s  (decode %.3fs, ttft %.3fs)\n", ar_tps, ar_decode_s, ar_ttft_s);
    printf("DSPARK %.2f tok/s  (decode %.3fs)  = %.3fx AR\n",
           spec_tps, stats.decode_s, ar_tps > 0 ? spec_tps / ar_tps : 0.0);

    // Machine-readable tail for the eval bot. LOSSLESS is emitted as 1/0 rather than YES/NO so a
    // parser that silently fails to find it reads as 0 (reject) and not as a pass -- the same
    // fail-closed shape the modelopt bot uses for a missing decode metric. It is a GATE, not a
    // score: a speculative decoder that is fast because it emits unverified tokens is not faster,
    // it is wrong, so throughput here is only meaningful conditional on this being 1.
    printf("METRIC AR_TPS %.4f\n", ar_tps);
    printf("METRIC DSPARK_TPS %.4f\n", spec_tps);
    printf("METRIC MEAN_ACCEPT %.4f\n", stats.mean_accept);
    printf("METRIC LOSSLESS %d\n", lossless ? 1 : 0);
    // How much evidence is behind that 1, so a consumer can tell "verified across N runs" from
    // "one run got lucky". 1 means single-shot, which only catches deterministic breakage.
    printf("METRIC LOSSLESS_RUNS %d\n", spec_reps_run > 0 ? spec_reps_run : 1);
    return 0;
}
