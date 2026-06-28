// thermal_sweep — force each ThermalGovernor mode and measure the trade-off it produces:
// decode throughput (tok/s) vs GPU power (W) and temperature (°C). Demonstrates that pacing
// trades speed for lower heat/watts WITHOUT changing the output: the generated token ids are
// asserted identical across every mode (only the timing differs).
//
// Usage: thermal_sweep <model.gguf> [n_tokens] [id0 id1 ...]

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/thermal_governor.h"
#include "sparkinfer/gpu_stats.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>

using G = sparkinfer::ThermalGovernor;

struct Reading { double tps; int peak_temp, avg_temp, peak_pw, avg_pw; uint64_t throttled; };

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model.gguf> [n_tokens] [id0 ...]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string path = argv[1];
    const int n_tokens = argc > 2 ? atoi(argv[2]) : 256;
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) prompt.push_back(atoi(argv[i]));
    if (prompt.empty()) prompt = {9707, 11, 1879, 0};   // a fixed, arbitrary prompt

    sparkinfer::GGUF g;
    if (!g.open(path)) { printf("[FAIL] cannot open %s\n", path.c_str()); return 1; }
    const char* A = "qwen3moe.";
    auto mi = [&](const std::string& k, long d){ return (int)g.meta_int(A + k, d); };
    sparkinfer::Qwen35Config cfg;
    cfg.n_layers=mi("block_count",48); cfg.hidden=mi("embedding_length",2048);
    cfg.n_q_heads=mi("attention.head_count",32); cfg.n_kv_heads=mi("attention.head_count_kv",4);
    cfg.head_dim=mi("attention.key_length",128); cfg.n_experts=mi("expert_count",128);
    cfg.top_k=mi("expert_used_count",8); cfg.moe_ffn=mi("expert_feed_forward_length",768);
    cfg.rope_theta=(float)g.meta_float(std::string(A)+"rope.freq_base",1e6);
    cfg.rms_eps=(float)g.meta_float(std::string(A)+"attention.layer_norm_rms_epsilon",1e-6);
    cfg.eos_id=(int)g.meta_int("tokenizer.ggml.eos_token_id",151645); cfg.n_shared=0;
    const sparkinfer::GGUFTensor* emb=g.tensor("token_embd.weight");
    cfg.vocab = emb ? (int)emb->dims[1] : 151936; cfg.max_seq=2048;

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers=cfg.n_layers; kvc.num_kv_heads=cfg.n_kv_heads; kvc.head_dim=cfg.head_dim; kvc.block_size=16;
    const size_t epb=(size_t)16*cfg.n_kv_heads*cfg.head_dim, blocks=(cfg.max_seq+15)/16+8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers*2*epb*2*blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts=cfg.n_experts; mc.top_k=cfg.top_k; mc.hidden_dim=cfg.hidden; mc.ffn_dim=cfg.moe_ffn; mc.num_layers=cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading GGUF ...\n");
    if (!model.load_gguf(path)) { printf("[FAIL] load_gguf\n"); return 1; }
    printf("loaded. %s\n", sparkinfer::query_gpu_stats().str().c_str());

    // One untimed run to warm caches / reach steady clocks before the first measured mode.
    model.generate(prompt, 16);

    const struct { const char* name; G::Mode m; } MODES[] = {
        {"turbo",     G::Mode::Turbo},
        {"balanced",  G::Mode::Balanced},
        {"safe",      G::Mode::Safe},
        {"emergency", G::Mode::Emergency},
    };

    std::vector<int> baseline;   // turbo output — every other mode must match it
    Reading rd[4];
    printf("\nsweeping %d modes × %d tokens (prompt %zu) ...\n", 4, n_tokens, prompt.size());

    for (int k = 0; k < 4; k++) {
        G::Config tc; tc.enabled = true; tc.forced = true; tc.forced_mode = MODES[k].m;
        G gov(tc);

        // Background sampler: GPU power + temperature every 100 ms while this mode decodes.
        std::atomic<bool> run{true};
        long sum_pw=0, sum_t=0, cnt=0; int pk_pw=0, pk_t=0;
        std::thread sampler([&]{
            while (run.load(std::memory_order_relaxed)) {
                auto s = sparkinfer::query_gpu_stats();
                if (s.valid) {
                    if (s.power_w >= 0) { sum_pw += s.power_w; if (s.power_w > pk_pw) pk_pw = s.power_w; }
                    if (s.temp_c  >= 0) { sum_t  += s.temp_c;  if (s.temp_c  > pk_t)  pk_t  = s.temp_c; cnt++; }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> out = model.generate(prompt, n_tokens, &gov);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        run.store(false, std::memory_order_relaxed); sampler.join();

        double secs = std::chrono::duration<double>(t1 - t0).count();
        rd[k] = { out.size() / secs,
                  pk_t, cnt ? (int)(sum_t/cnt) : -1,
                  pk_pw, cnt ? (int)(sum_pw/cnt) : -1,
                  gov.throttled_tokens() };
        if (k == 0) baseline = out;
        const bool same = (out == baseline);
        printf("  %-9s : %6.1f tok/s · temp %d°C peak / %d°C avg · power %d W peak / %d W avg · "
               "throttled %llu · output %s\n",
               MODES[k].name, rd[k].tps, rd[k].peak_temp, rd[k].avg_temp, rd[k].peak_pw, rd[k].avg_pw,
               (unsigned long long)rd[k].throttled, same ? "IDENTICAL" : "*** DIFFERS ***");
        if (!same) { printf("[FAIL] mode %s changed the output — pacing must be accuracy-preserving\n", MODES[k].name); return 1; }
    }

    // Summary table relative to turbo.
    printf("\n=== thermal sweep summary (vs turbo) ===\n");
    printf("%-9s  %8s  %8s  %8s  %9s  %9s\n", "mode", "tok/s", "x speed", "avg W", "avg °C", "Δ°C peak");
    for (int k = 0; k < 4; k++)
        printf("%-9s  %8.1f  %7.2fx  %8d  %9d  %+9d\n",
               MODES[k].name, rd[k].tps,
               rd[0].tps > 0 ? rd[k].tps / rd[0].tps : 0.0,
               rd[k].avg_pw, rd[k].avg_temp, rd[k].peak_temp - rd[0].peak_temp);
    printf("\nall modes produced IDENTICAL token ids — pacing trades only speed for heat/power.\n");
    return 0;
}
