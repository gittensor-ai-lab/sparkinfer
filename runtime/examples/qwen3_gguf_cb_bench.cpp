// Concurrent continuous-batching bench (vLLM-style mixed load).
// Measures aggregate decode tok/s and decode ITPS when a long prefill arrives
// while shorter decode streams are in flight.
//
// Usage:
//   qwen3_gguf_cb_bench <model.gguf> [concurrency=4] [prompt_len=128] [max_new=64] [long_prefill=4096]
//
// Env:
//   SPARKINFER_SCHED_POLICY=continuous|chunked|priority
//   SPARKINFER_BATCH_TOKENS=64
//   SPARKINFER_PREFILL_CHUNK_TOKENS=512

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/inference_engine.h"
#include "sparkinfer/scheduler.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static sparkinfer::SchedulePolicy parse_policy() {
    const char* p = getenv("SPARKINFER_SCHED_POLICY");
    if (!p) return sparkinfer::SchedulePolicy::CONTINUOUS_BATCHING;
    if (strncmp(p, "chunk", 5) == 0) return sparkinfer::SchedulePolicy::CHUNKED_PREFILL;
    if (p[0] == 'p' || p[0] == 'P') return sparkinfer::SchedulePolicy::PRIORITY;
    return sparkinfer::SchedulePolicy::CONTINUOUS_BATCHING;
}

static const char* policy_name(sparkinfer::SchedulePolicy p) {
    switch (p) {
        case sparkinfer::SchedulePolicy::CHUNKED_PREFILL: return "chunked";
        case sparkinfer::SchedulePolicy::PRIORITY: return "priority";
        default: return "continuous";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf> [concurrency] [prompt_len] [max_new] [long_prefill]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n");
        return 0;
    }

    const std::string path = argv[1];
    const int concurrency = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
    const int prompt_len = argc > 3 ? std::max(8, atoi(argv[3])) : 128;
    const int max_new = argc > 4 ? std::max(1, atoi(argv[4])) : 64;
    const int long_prefill = argc > 5 ? std::max(prompt_len, atoi(argv[5])) : 4096;
    const int batch_tokens = [] {
        const char* e = getenv("SPARKINFER_BATCH_TOKENS");
        return e ? std::max(1, atoi(e)) : 64;
    }();
    const auto policy = parse_policy();

    sparkinfer::GGUF g;
    if (!g.open(path)) {
        printf("[FAIL] cannot open %s\n", path.c_str());
        return 1;
    }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq = std::max(cfg.max_seq, long_prefill + max_new + 64);
    cfg.eos_id = -1;  // force full max_new for stable throughput accounting

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    {
        const char* e = getenv("SPARKINFER_KV_INT8");
        kvc.int8_kv = e ? (e[0] != '0') : (cfg.hybrid ? (cfg.max_seq >= 4096) : true);
    }
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    // Pool for concurrency streams + one long prefill.
    const size_t blocks = (size_t)(concurrency + 1) * ((cfg.max_seq + 15) / 16 + 4);
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading %s (policy=%s batch=%d chunk=%s) ...\n", path.c_str(), policy_name(policy),
           batch_tokens, getenv("SPARKINFER_PREFILL_CHUNK_TOKENS")
                             ? getenv("SPARKINFER_PREFILL_CHUNK_TOKENS")
                             : "512");
    if (!model.load_gguf(path)) {
        printf("[FAIL] load_gguf\n");
        return 1;
    }

    sparkinfer::ContinuousBatchEngine batch(&model, &kv, batch_tokens, policy);

    std::vector<int> short_prompt(prompt_len, 42);
    for (int i = 0; i < prompt_len; i++) short_prompt[i] = 100 + (i % 50);
    std::vector<int> long_prompt(long_prefill, 7);
    for (int i = 0; i < long_prefill; i++) long_prompt[i] = 200 + (i % 70);

    std::atomic<int> decode_tokens{0};
    std::atomic<int> first_decode_tokens{0};
    std::atomic<bool> long_started{false};
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    std::atomic<int64_t> sum_decode_gap_us{0};
    std::atomic<int64_t> max_decode_gap_us{0};
    std::atomic<int> n_decode_gaps{0};
    std::atomic<int64_t> long_ttft_us{-1};   // submit -> first long-request token (prefill done)
    std::atomic<int64_t> long_done_us{-1};   // submit -> long request complete

    auto run_stream = [&](const std::vector<int>& prompt, int n_new, bool track_itl,
                          bool measure_long) {
        sparkinfer::ContinuousBatchEngine::Request req;
        req.prompt = prompt;
        req.max_new_tokens = n_new;
        clock::time_point prev{};
        bool have_prev = false;
        const auto t_submit = clock::now();
        auto on_tok = [&](int) {
            decode_tokens.fetch_add(1, std::memory_order_relaxed);
            if (measure_long && long_ttft_us.load(std::memory_order_relaxed) < 0) {
                const auto us =
                    std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_submit)
                        .count();
                long_ttft_us.store(us, std::memory_order_relaxed);
            }
            if (!track_itl) return;
            const auto now = clock::now();
            if (have_prev) {
                const auto us =
                    std::chrono::duration_cast<std::chrono::microseconds>(now - prev).count();
                sum_decode_gap_us.fetch_add(us, std::memory_order_relaxed);
                n_decode_gaps.fetch_add(1, std::memory_order_relaxed);
                int64_t prev_max = max_decode_gap_us.load(std::memory_order_relaxed);
                while (us > prev_max &&
                       !max_decode_gap_us.compare_exchange_weak(prev_max, us,
                                                                std::memory_order_relaxed)) {
                }
                if (long_started.load(std::memory_order_relaxed))
                    first_decode_tokens.fetch_add(1, std::memory_order_relaxed);
            }
            prev = now;
            have_prev = true;
        };
        auto r = batch.complete_streaming(req, on_tok);
        if (measure_long) {
            const auto us =
                std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_submit)
                    .count();
            long_done_us.store(us, std::memory_order_relaxed);
        }
        if (!r.error.empty()) fprintf(stderr, "[warn] request error: %s\n", r.error.c_str());
    };

    // Warm one short request so weights/caches are resident.
    run_stream(short_prompt, 8, false, false);
    decode_tokens.store(0);
    sum_decode_gap_us.store(0);
    max_decode_gap_us.store(0);
    n_decode_gaps.store(0);
    first_decode_tokens.store(0);
    long_ttft_us.store(-1);
    long_done_us.store(-1);

    std::vector<std::thread> workers;
    workers.reserve((size_t)concurrency + 1);
    const auto t_run = clock::now();
    for (int i = 0; i < concurrency; i++) {
        workers.emplace_back([&, i] {
            run_stream(short_prompt, max_new, /*track_itl=*/true, /*measure_long=*/false);
        });
    }
    // Inject long prefill shortly after decode streams start.
    workers.emplace_back([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        long_started.store(true, std::memory_order_relaxed);
        run_stream(long_prompt, 8, false, /*measure_long=*/true);
    });
    for (auto& t : workers) t.join();
    const auto t1 = clock::now();

    const double wall_s = std::chrono::duration<double>(t1 - t_run).count();
    const int toks = decode_tokens.load();
    const int gaps = n_decode_gaps.load();
    const double itl_ms =
        gaps > 0 ? (1e-3 * (double)sum_decode_gap_us.load() / (double)gaps) : 0.0;
    const double max_itl_ms = 1e-3 * (double)max_decode_gap_us.load();
    const double long_ttft_s =
        long_ttft_us.load() > 0 ? 1e-6 * (double)long_ttft_us.load() : 0.0;
    const double long_pp =
        long_ttft_s > 0 ? (double)long_prefill / long_ttft_s : 0.0;
    printf("cb_bench policy=%s concurrency=%d prompt=%d max_new=%d long_prefill=%d\n",
           policy_name(policy), concurrency, prompt_len, max_new, long_prefill);
    printf("wall_s=%.3f decode_tokens=%d agg_tok_s=%.1f mean_itl_ms=%.2f max_itl_ms=%.2f\n",
           wall_s, toks, wall_s > 0 ? toks / wall_s : 0.0, itl_ms, max_itl_ms);
    printf("long_ttft_s=%.3f long_prefill_pp=%.1f tokens_while_long_active=%d\n", long_ttft_s,
           long_pp, first_decode_tokens.load());
    (void)t0;
    (void)long_done_us;
    return 0;
}
