// Decode-throughput benchmark for the sparkinfer Qwen3 runtime.
// Reports steady-state single-stream generation tokens/sec, to compare against
// llama.cpp's `llama-bench` tg number on the same model + GPU.
//
// Single context:
//   qwen3_gguf_bench <model.gguf | compressed-tensors dir | weight_dir> [n_tokens] [context_tokens]
//
// Multi-context sweep (one load, many contexts) — set SPARKINFER_BENCH_SWEEP_CTXS:
//   qwen3_gguf_bench <model> [n_tokens] sweep
//   SPARKINFER_BENCH_SWEEP_CTXS=0,4096,32768,65536
//
// Sweep prints human-readable lines per context plus a final SWEEP_JSON line.

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <memory>

static std::vector<int> parse_ctx_list(const char* csv) {
    std::vector<int> out;
    if (!csv || !*csv) return out;
    const char* p = csv;
    while (*p) {
        char* end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out.push_back((int)v);
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct SweepRow {
    int ctx = 0;
    double decode_tps = 0.0;
    double prefill_pp = 0.0;
};

static double median_val(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[(v.size() - 1) / 2];
}

static void print_bench_block(const sparkinfer::Qwen35Config& cfg, const char* quant_label,
                              size_t vram_used, int n_tokens, const SweepRow& row) {
    printf("\n=== sparkinfer bench (%s) sweep ctx=%d ===\n",
           quant_label, row.ctx);
    printf("model        : %d layers, %d experts top-%d\n",
           cfg.n_layers, cfg.n_experts, cfg.top_k);
    printf("VRAM used    : %.1f GB\n", vram_used / 1e9);
    printf("max seq      : %d\n", cfg.max_seq);
    printf("decode tg    : %.2f tok/s  (n=%d, ctx=%d, bs=1)\n",
           row.decode_tps, n_tokens, row.ctx);
    if (row.ctx > 0)
        printf("prefill pp   : %.2f tok/s  (ctx=%d, sequential KV fill)\n",
               row.prefill_pp, row.ctx);
}

struct BenchSession {
    sparkinfer::Qwen35Config cfg{};
    QwenCheckpointKind kind = QwenCheckpointKind::Gguf;
    std::unique_ptr<sparkinfer::Runtime> rt;
    std::unique_ptr<sparkinfer::KVCacheManager> kv;
    std::unique_ptr<sparkinfer::moe::MoEEngine> engine;
    std::unique_ptr<sparkinfer::Qwen35Model> model;
    size_t vram_used = 0;
};

static bool init_session(BenchSession& s, const std::string& path, int max_ctx, int n_tokens) {
    sparkinfer::GGUF g;
    std::string cperr;
    if (!qwen_checkpoint_open(path, s.cfg, g, s.kind, cperr)) {
        printf("[FAIL] %s\n", cperr.c_str());
        return false;
    }
    if (s.kind == QwenCheckpointKind::LegacyWeightDir) {
        std::ifstream f(path + "/config.txt");
        std::string line;
        std::unordered_map<std::string, std::string> m;
        while (std::getline(f, line)) {
            auto p = line.find('=');
            if (p != std::string::npos) m[line.substr(0, p)] = line.substr(p + 1);
        }
        auto gi = [&](const char* k, int d) {
            auto it = m.find(k); return it == m.end() ? d : atoi(it->second.c_str());
        };
        auto gf = [&](const char* k, float d) {
            auto it = m.find(k); return it == m.end() ? d : (float)atof(it->second.c_str());
        };
        s.cfg.vocab = gi("vocab", 151936); s.cfg.hidden = gi("hidden", 2048);
        s.cfg.n_layers = gi("n_layers", 48); s.cfg.n_q_heads = gi("n_q_heads", 32);
        s.cfg.n_kv_heads = gi("n_kv_heads", 4); s.cfg.head_dim = gi("head_dim", 128);
        s.cfg.n_experts = gi("n_experts", 128); s.cfg.top_k = gi("top_k", 8);
        s.cfg.n_shared = gi("n_shared", 0); s.cfg.moe_ffn = gi("moe_ffn", 768);
        s.cfg.rope_theta = gf("rope_theta", 1e6f); s.cfg.rms_eps = gf("rms_eps", 1e-6f);
    }
    s.cfg.max_seq = std::max(2048, max_ctx + n_tokens + 16);
    if (const char* e = getenv("SPARKINFER_BENCH_MAX_SEQ")) {
        int v = atoi(e);
        if (v > s.cfg.max_seq) s.cfg.max_seq = v;
    }

    s.rt = sparkinfer::Runtime::create({});
    s.rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = s.cfg.n_layers;
    kvc.num_kv_heads = s.cfg.n_kv_heads;
    kvc.head_dim = s.cfg.head_dim;
    kvc.block_size = 16;
    const char* e8 = getenv("SPARKINFER_KV_INT8");
    kvc.int8_kv = e8 ? (e8[0] != '0') : (max_ctx >= 4096);
    // Hybrid: only every full_attn_interval-th layer owns KV; the rest are Gated-DeltaNet and keep
    // their state in the recurrent/conv buffers. Without this the pool reserves sub-pools for all
    // 64 of Qwen3.8's layers when 16 are used. SPARKINFER_KV_COMPACT=0 restores the full pool.
    {
        const char* kc = getenv("SPARKINFER_KV_COMPACT");
        if (!(kc && kc[0] == '0') && s.cfg.hybrid && s.cfg.full_attn_interval > 1 &&
            s.cfg.n_layers % s.cfg.full_attn_interval == 0)
            kvc.attn_every = s.cfg.full_attn_interval;
    }
    const size_t epb = (size_t)16 * s.cfg.n_kv_heads * s.cfg.head_dim;
    const size_t blocks = (s.cfg.max_seq + 15) / 16 + 8;
    s.kv = std::make_unique<sparkinfer::KVCacheManager>(
        kvc, (size_t)s.cfg.n_layers * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = s.cfg.n_experts; mc.top_k = s.cfg.top_k;
    mc.hidden_dim = s.cfg.hidden; mc.ffn_dim = s.cfg.moe_ffn;
    mc.num_layers = s.cfg.n_layers;
    s.engine = sparkinfer::moe::MoEEngine::create(mc);
    s.model = std::make_unique<sparkinfer::Qwen35Model>(s.cfg, s.kv.get(), s.engine.get());

    printf("loading %s (%s) ...\n", path.c_str(), qwen_checkpoint_kind_label(s.kind));
    bool ok = qwen_checkpoint_load(*s.model, path, s.kind);
    if (!ok) { printf("[FAIL] load\n"); return false; }
    size_t freeb = 0, totb = 0;
    cudaMemGetInfo(&freeb, &totb);
    s.vram_used = totb - freeb;
    return true;
}

static int run_single(const std::string& path, int n_tokens, int context_tokens) {
    BenchSession s;
    if (!init_session(s, path, context_tokens, n_tokens)) return 1;

    const int suffix_len = (context_tokens > 0 && getenv("SPARKINFER_BENCH_PREFIX_TTFT"))
        ? std::max(0, atoi(getenv("SPARKINFER_BENCH_PREFIX_TTFT")))
        : 0;
    if (suffix_len > 0 && context_tokens > suffix_len) {
        std::vector<int> prefix((size_t)(context_tokens - suffix_len), 100);
        std::vector<int> full((size_t)context_tokens, 100);
        s.model->clear_prefix_cache();
        double cold = s.model->bench_ttft(full);
        if (!s.model->cache_prefix(prefix)) {
            printf("[FAIL] cache_prefix(%d tokens)\n", (int)prefix.size());
            return 1;
        }
        double warm = s.model->bench_ttft(full);
        s.model->clear_prefix_cache();
        printf("prefix_ttft  : cold %.3f s (full %d tok) · warm %.3f s (reuse %d + suffix %d)\n",
               cold, context_tokens, warm, (int)prefix.size(), suffix_len);
        if (warm > 0. && cold > 0.)
            printf("prefix_speedup: %.1fx TTFT on cached prefix\n", cold / warm);
    } else {
        const bool bench_ttft = context_tokens > 0 &&
            (getenv("SPARKINFER_BENCH_TTFT") || context_tokens >= 512);
        if (bench_ttft) {
            std::vector<int> prompt((size_t)context_tokens, 100);
            double ttft = s.model->bench_ttft(prompt);
            if (ttft > 0.)
                printf("ttft         : %.3f s  (%.1f tok/s prefill, prompt=%d)\n",
                       ttft, context_tokens / ttft, context_tokens);
        }
    }

    auto bench = s.model->bench_decode(8, n_tokens, context_tokens);
    auto gpu = sparkinfer::query_gpu_stats();
    printf("\n=== sparkinfer bench (%s) ===\n", qwen_checkpoint_kind_label(s.kind));
    printf("model        : %s  (%d layers, %d experts top-%d)\n",
           qwen3_model_label(s.cfg), s.cfg.n_layers, s.cfg.n_experts, s.cfg.top_k);
    printf("VRAM used    : %.1f GB\n", s.vram_used / 1e9);
    printf("max seq      : %d\n", s.cfg.max_seq);
    printf("decode tg    : %.2f tok/s  (%.1f ms/token, n=%d, ctx=%d, bs=1)\n",
           bench.decode_tps, bench.decode_tps > 0 ? 1000.0 / bench.decode_tps : 0.0,
           n_tokens, context_tokens);
    if (context_tokens > 0)
        printf("prefill pp   : %.2f tok/s  (ctx=%d, sequential KV fill)\n",
               bench.prefill_pp, context_tokens);
    if (gpu.valid && gpu.temp_c >= 0) {
        printf("GPU          : %d°C", gpu.temp_c);
        if (gpu.power_w >= 0) printf(" · %d W", gpu.power_w);
        if (gpu.sm_clock_mhz >= 0) printf(" · %d MHz", gpu.sm_clock_mhz);
        printf(" (peak under load)\n");
    }
    return 0;
}

static int run_sweep(const std::string& path, int n_tokens, const std::vector<int>& ctxs) {
    if (ctxs.empty()) { fprintf(stderr, "[FAIL] empty sweep ctx list\n"); return 1; }
    const int max_ctx = *std::max_element(ctxs.begin(), ctxs.end());
    BenchSession s;
    if (!init_session(s, path, max_ctx, n_tokens)) return 1;

    int reps = 1;
    if (const char* er = getenv("SPARKINFER_BENCH_SWEEP_REPS")) reps = std::max(1, atoi(er));

    printf(">> sweep: %zu contexts, one model load (max_ctx=%d, n=%d, reps=%d)\n",
           ctxs.size(), max_ctx, n_tokens, reps);

    // Clock / cache warmup (not scored).
    s.model->bench_decode(8, 64, 0);

    std::vector<SweepRow> rows;
    rows.reserve(ctxs.size());
    for (int ctx : ctxs) {
        std::vector<double> dec, pp;
        for (int r = 0; r < reps; r++) {
            auto b = s.model->bench_decode(8, n_tokens, ctx);
            dec.push_back(b.decode_tps);
            pp.push_back(b.prefill_pp);
        }
        SweepRow row;
        row.ctx = ctx;
        row.decode_tps = median_val(dec);
        row.prefill_pp = ctx > 0 ? median_val(pp) : 0.0;
        rows.push_back(row);
        print_bench_block(s.cfg, qwen_checkpoint_kind_label(s.kind), s.vram_used, n_tokens, row);
    }

    auto gpu = sparkinfer::query_gpu_stats();
    if (gpu.valid && gpu.temp_c >= 0) {
        printf("GPU          : %d°C", gpu.temp_c);
        if (gpu.power_w >= 0) printf(" · %d W", gpu.power_w);
        if (gpu.sm_clock_mhz >= 0) printf(" · %d MHz", gpu.sm_clock_mhz);
        printf(" (peak under load)\n");
    }

    printf("SWEEP_JSON {");
    for (size_t i = 0; i < rows.size(); i++) {
        if (i) printf(",");
        printf("\"%d\":{\"decode_tps\":%.4f,\"prefill_pp\":%.4f}",
               rows[i].ctx, rows[i].decode_tps, rows[i].prefill_pp);
    }
    printf("}\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf|compressed-tensors dir|weight_dir> [n_tokens] [context_tokens|sweep]\n", argv[0]);
        printf("  sweep: SPARKINFER_BENCH_SWEEP_CTXS=0,4096,... %s <model> [n_tokens] sweep\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n");
        return 0;
    }
    const std::string path = argv[1];
    const int n_tokens = argc > 2 ? atoi(argv[2]) : 64;
    const bool sweep_mode = (argc > 3 && std::string(argv[3]) == "sweep") ||
                            (getenv("SPARKINFER_BENCH_SWEEP_CTXS") && argc <= 3);
    if (sweep_mode) {
        const char* csv = getenv("SPARKINFER_BENCH_SWEEP_CTXS");
        if (!csv || !*csv) {
            fprintf(stderr, "[FAIL] sweep requires SPARKINFER_BENCH_SWEEP_CTXS\n");
            return 1;
        }
        return run_sweep(path, n_tokens, parse_ctx_list(csv));
    }
    const int context_tokens = argc > 3 ? atoi(argv[3]) : 0;
    return run_single(path, n_tokens, context_tokens);
}
