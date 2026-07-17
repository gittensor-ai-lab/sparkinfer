// MTP correctness gate: speculative output vs AR, draft-head top-1 / KL vs main model.
//
// Usage: qwen3_gguf_mtp_check <model.gguf> <max_new> <id0> <id1> ...
// Env:   SPARKINFER_MTP=1  SPARKINFER_MTP_DRAFT_MAX=3  SPARKINFER_MTP_FAST=0
//
// Reports:
//   SPEC_AGREE   greedy generate MTP-off vs MTP-on (want 1.0000)
//   DRAFT_TOP1   MTP first draft == main verify argmax
//   MAIN_TOP1    main argmax == teacher next (trunk sanity)
//   METRIC       machine-readable summary line

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s <model.gguf> <max_new> <id0> [id1 ...]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string path = argv[1];
    const int max_new = atoi(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) prompt.push_back(atoi(argv[i]));
    if (prompt.empty()) { printf("[FAIL] empty prompt\n"); return 1; }

    sparkinfer::GGUF g;
    if (!g.open(path)) { printf("[FAIL] cannot open %s\n", path.c_str()); return 1; }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq = std::max(2048, (int)prompt.size() + max_new + 64);
    if (cfg.n_nextn_layers <= 0) {
        printf("[FAIL] model has no MTP / NextN head (n_nextn_layers=0)\n");
        return 1;
    }

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    { const char* e = getenv("SPARKINFER_KV_INT8");
      kvc.int8_kv = e ? (e[0] != '0') : (cfg.hybrid ? false : true); }
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_gguf(path)) { printf("[FAIL] load_gguf\n"); return 1; }

    // --- 1) Speculative vs AR greedy output ---
    model.set_mtp_decode(false);
    std::vector<int> ar_out = model.generate(prompt, max_new, nullptr, nullptr);
    model.set_mtp_decode(true);
    std::vector<int> mtp_out = model.generate(prompt, max_new, nullptr, nullptr);

    int spec_match = 0;
    const int spec_n = (int)std::min(ar_out.size(), mtp_out.size());
    for (int i = 0; i < spec_n; i++)
        if (ar_out[i] == mtp_out[i]) spec_match++;
    const bool spec_len_ok = ar_out.size() == mtp_out.size();
    const double spec_agree = spec_n > 0 ? (double)spec_match / (double)spec_n : 1.0;

    // Build teacher-forced sequence: prompt + AR greedy continuation
    std::vector<int> teacher = prompt;
    teacher.insert(teacher.end(), ar_out.begin(), ar_out.end());

    int warmup = 2;
    if (const char* w = getenv("SPARKINFER_MTP_WARMUP")) warmup = atoi(w);

    auto dm = model.mtp_draft_check(teacher, warmup);
    const double main_top1 = dm.positions > 0 ? (double)dm.main_top1 / dm.positions : 1.0;
    const double draft_top1 = dm.positions > 0 ? (double)dm.draft_top1 / dm.positions : 0.0;

    printf("=== MTP correctness (Qwythos MTP) ===\n");
    printf("positions (draft) : %d  (warmup=%d)\n", dm.positions, warmup);
    printf("SPEC_AGREE        : %d/%d = %.4f  len_match=%s  (want 1.0000)\n",
           spec_match, spec_n, spec_agree, spec_len_ok ? "yes" : "NO");
    printf("MAIN_TOP1         : %d/%d = %.4f  (main greedy vs teacher)\n",
           dm.main_top1, dm.positions, main_top1);
    printf("DRAFT_TOP1        : %d/%d = %.4f  (MTP draft vs main verify)\n",
           dm.draft_top1, dm.positions, draft_top1);
    printf("mean KL(main||mtp): %.4f nats  (informational: draft-head quality, not correctness)\n", dm.mean_kl);
    printf("METRIC spec_agree=%.6f draft_top1=%.6f main_top1=%.6f kl=%.6f\n",
           spec_agree, draft_top1, main_top1, dm.mean_kl);

    // Correctness gate: the target verifies every draft exactly, so generation output is
    // guaranteed AR-identical iff SPEC_AGREE holds. DRAFT_TOP1 gates the head being wired
    // well enough to be worth running (it only affects speed); KL is reported for tracking.
    const bool pass = spec_agree >= 0.999 && spec_len_ok && draft_top1 >= 0.50;
    printf("VERDICT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
