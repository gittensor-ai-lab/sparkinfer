// Run a real Qwen3-MoE model directly from a GGUF file (native load).
// Dense weights are dequantized to bf16 at load; expert weights stay quantized
// in VRAM (Q4_K/Q6_K) and are dequantized per-layer at decode time — so the
// resident footprint is the Q4_K_M size, not the bf16 expansion.
//
// Usage: qwen3_gguf_generate <model.gguf> <max_new> <id0> <id1> ...
// (tokenize/detokenize with tools/run_qwen3.py)

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/thermal_governor.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <model.gguf> <max_new> <id0> [id1 ...]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string path = argv[1];
    const int max_new = atoi(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) prompt.push_back(atoi(argv[i]));

    // read architecture from GGUF metadata
    sparkinfer::GGUF g;
    if (!g.open(path)) { printf("[FAIL] cannot open %s\n", path.c_str()); return 1; }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq    = std::max(2048, (int)prompt.size() + max_new + 16);
    printf("arch: %s, %d layers, hidden %d, %dQ/%dKV hd%d, %d experts top-%d, ffn %d, vocab %d\n",
           qwen3_model_label(cfg), cfg.n_layers, cfg.hidden, cfg.n_q_heads,
           cfg.n_kv_heads, cfg.head_dim, cfg.n_experts, cfg.top_k, cfg.moe_ffn, cfg.vocab);

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    // int8 KV is the Qwen3-MoE head_dim=128 tensor-core path; Qwen3.6 attention (gated, head_dim=256) writes bf16 KV.
    { const char* e = getenv("SPARKINFER_KV_INT8");   // hybrid: int8 KV when prompt length >= 4k
      kvc.int8_kv = e ? (e[0] != '0') : (cfg.hybrid ? ((argc - 3) >= 4096) : true); }
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading GGUF (dense->bf16, experts kept quantized) ...\n");
    if (!model.load_gguf(path)) { printf("[FAIL] load_gguf\n"); return 1; }
    {
        auto g = sparkinfer::query_gpu_stats();
        printf("loaded. GPU: %s. generating %d tokens from %zu prompt tokens\n",
               g.str().c_str(), max_new, prompt.size());
    }

    // GPU observability: poll heat/VRAM/power on a background thread during decode and keep the
    // PEAK, so the run reports the hottest the workload drove the device (and whether it throttled).
    std::atomic<bool> sampling{true};
    sparkinfer::GpuStats peak;
    std::thread sampler([&] {
        while (sampling.load(std::memory_order_relaxed)) {
            auto s = sparkinfer::query_gpu_stats();
            if (s.valid) {
                peak.valid = true;
                if (s.temp_c          > peak.temp_c)          peak.temp_c          = s.temp_c;
                if (s.power_w          > peak.power_w)          peak.power_w          = s.power_w;
                if (s.sm_clock_mhz     > peak.sm_clock_mhz)     peak.sm_clock_mhz     = s.sm_clock_mhz;
                if (s.vram_used_bytes  > peak.vram_used_bytes)  peak.vram_used_bytes  = s.vram_used_bytes;
                peak.vram_total_bytes = s.vram_total_bytes;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    // Thermally-adaptive decode pacing (opt-in via SPARKINFER_THERMAL=1). Accuracy-preserving —
    // it only slows token emission when the GPU runs hot. Thresholds/paces overridable via env.
    auto envi = [](const char* k, int d){ const char* v = getenv(k); return v ? atoi(v) : d; };
    sparkinfer::ThermalGovernor::Config tg;
    tg.enabled         = envi("SPARKINFER_THERMAL", 0) != 0;
    tg.balanced_c      = envi("SPARKINFER_THERMAL_BALANCED_C", tg.balanced_c);
    tg.safe_c          = envi("SPARKINFER_THERMAL_SAFE_C",     tg.safe_c);
    tg.emergency_c     = envi("SPARKINFER_THERMAL_EMERGENCY_C", tg.emergency_c);
    tg.log_transitions = true;
    sparkinfer::ThermalGovernor gov(tg);
    if (tg.enabled)
        printf("thermal: ON (turbo<%d°C, balanced≥%d, safe≥%d, emergency≥%d)\n",
               tg.balanced_c, tg.balanced_c, tg.safe_c, tg.emergency_c);

    if (const char* pl = getenv("SPARKINFER_PREFIX_CACHE_LEN")) {
        int n = atoi(pl);
        if (n > 0 && n < (int)prompt.size()) {
            std::vector<int> prefix(prompt.begin(), prompt.begin() + n);
            if (!model.cache_prefix(prefix))
                printf("[FAIL] cache_prefix(%d)\n", n);
            else
                printf("prefix_cache: installed %d / %zu prompt tokens\n", n, prompt.size());
        }
    }

    auto out = model.generate(prompt, max_new, tg.enabled ? &gov : nullptr);
    sampling.store(false, std::memory_order_relaxed);
    sampler.join();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("[FAIL] cuda: %s\n", cudaGetErrorString(e)); return 1; }
    printf("GPU peak under load: %s\n", peak.str().c_str());
    if (tg.enabled)
        printf("thermal: final mode=%s, peak %d°C, %llu/%d tokens throttled\n",
               sparkinfer::ThermalGovernor::mode_name(gov.mode()), gov.peak_temp_c(),
               (unsigned long long)gov.throttled_tokens(), (int)out.size());

    printf("OUTPUT_IDS:");
    for (int id : out) printf(" %d", id);
    printf("\n");
    return 0;
}
