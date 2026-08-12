// TTFT benchmark: LMCache-enabled (cache hit, repeated prefix) vs. baseline (full recompute) on
// the same prompt. Uses a synthetic random-weight model, same construction as
// runtime/tests/batch_engine_gpu_test.cpp / lmcache_e2e_gpu_test.cpp, sized larger here to give
// prefill a realistic compute cost -- benchmark timing depends on model *shape* (layers, hidden
// size, head counts), not trained weight *values*, so a same-shaped random model gives a
// meaningful TTFT comparison without needing a multi-GB real GGUF download.
//
// Requires SPARKINFER_LMCACHE_PYTHON / SPARKINFER_LMCACHE_BRIDGE_SCRIPT pointing at a real
// lmcache+torch venv (see bridge/README.md) -- exits early with a clear message if unset, same
// convention as lmcache_e2e_gpu_test.cpp's skip behavior.
#include "sparkinfer/inference_engine.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/lmcache_bridge_client.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/runtime.h"

#include <sys/wait.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

void* rand_bf16(size_t n, float s) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) {
        float f = s * (2.f * ((i * 2654435761u + 40503u) % 1000) / 1000.f - 1.f);
        uint32_t b;
        __builtin_memcpy(&b, &f, 4);
        h[i] = (uint16_t)(b >> 16);
    }
    void* d = nullptr;
    cudaMalloc(&d, n * sizeof(uint16_t));
    cudaMemcpy(d, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    return d;
}

void fill_weights(sparkinfer::Qwen35Weights& w, const sparkinfer::Qwen35Config& cfg) {
    const int H = cfg.hidden, Q = cfg.n_q_heads * cfg.head_dim, KV = cfg.n_kv_heads * cfg.head_dim;
    const int E = cfg.n_experts, F = cfg.moe_ffn;
    w.embed_tokens = rand_bf16((size_t)cfg.vocab * H, 1.f);
    w.final_norm = rand_bf16(H, 0.5f);
    w.lm_head = rand_bf16((size_t)H * cfg.vocab, 0.05f);
    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) {
        auto& lw = w.layers[l];
        lw.input_norm = rand_bf16(H, 0.5f);
        lw.wq = rand_bf16((size_t)H * Q, 0.04f);
        lw.wk = rand_bf16((size_t)H * KV, 0.04f);
        lw.wv = rand_bf16((size_t)H * KV, 0.04f);
        lw.wo = rand_bf16((size_t)Q * H, 0.04f);
        lw.q_norm = rand_bf16(cfg.head_dim, 0.5f);
        lw.k_norm = rand_bf16(cfg.head_dim, 0.5f);
        lw.post_attn_norm = rand_bf16(H, 0.5f);
        lw.router_w = rand_bf16((size_t)H * E, 0.1f);
        lw.gate = rand_bf16((size_t)E * H * F, 0.04f);
        lw.up = rand_bf16((size_t)E * H * F, 0.04f);
        lw.down = rand_bf16((size_t)E * F * H, 0.04f);
        lw.shared_gate = rand_bf16((size_t)H * F, 0.04f);
        lw.shared_up = rand_bf16((size_t)H * F, 0.04f);
        lw.shared_down = rand_bf16((size_t)F * H, 0.04f);
    }
}

pid_t spawn_sidecar(const std::string& python, const std::string& script,
                    const std::string& socket_path, const sparkinfer::Qwen35Config& cfg,
                    const sparkinfer::KVCacheManager& kv) {
    std::vector<std::string> args = {
        python, script,
        "--socket", socket_path,
        "--instance-id", "lmcache-bench",
        "--num-layers", std::to_string(cfg.n_layers),
        "--num-kv-heads", std::to_string(cfg.n_kv_heads),
        "--head-dim", std::to_string(cfg.head_dim),
        "--block-size", std::to_string(kv.block_size()),
        "--elem-bytes", std::to_string(kv.int8_kv() ? 1 : 2),
        "--model-name", "lmcache-bench-model",
        "--log-level", "WARNING",
    };
    if (kv.int8_kv()) args.push_back("--int8-kv");
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execvp(python.c_str(), argv.data());
        fprintf(stderr, "[lmcache_bench] exec failed: %s\n", strerror(errno));
        _exit(127);
    }
    return pid;
}

void terminate_sidecar(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 100; i++) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) return;
        usleep(100000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("no CUDA device\n");
        return 1;
    }
    const char* python_env = getenv("SPARKINFER_LMCACHE_PYTHON");
    const char* script_env = getenv("SPARKINFER_LMCACHE_BRIDGE_SCRIPT");
    if (!python_env || !python_env[0] || !script_env || !script_env[0]) {
        printf("SPARKINFER_LMCACHE_PYTHON / SPARKINFER_LMCACHE_BRIDGE_SCRIPT not set -- see "
               "bridge/README.md\n");
        return 1;
    }
    const int prompt_len = getenv("LMCACHE_BENCH_PROMPT_LEN") ? atoi(getenv("LMCACHE_BENCH_PROMPT_LEN")) : 4096;

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    // Larger than the correctness test's tiny model -- realistic-enough shape for prefill
    // compute cost to be meaningful (this is a mid-size dense-ish MoE config, not tuned to match
    // any specific real model, just big enough that a multi-thousand-token prefill takes a
    // measurable amount of wall-clock time).
    sparkinfer::Qwen35Config cfg;
    cfg.vocab = 32000;
    cfg.hidden = 4096;
    cfg.n_layers = 8;
    cfg.n_q_heads = 32;
    cfg.n_kv_heads = 4;
    cfg.head_dim = 128;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.n_shared = 1;
    cfg.moe_ffn = 1024;
    cfg.max_seq = prompt_len + 256;
    cfg.eos_id = -1;

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    sparkinfer::KVCacheManager kv(kvc, 4ull * 1024 * 1024 * 1024);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    sparkinfer::Qwen35Weights w;
    printf("[lmcache_bench] building synthetic model (hidden=%d layers=%d) ...\n", cfg.hidden, cfg.n_layers);
    fill_weights(w, cfg);
    model.set_weights(w);

    const std::string socket_path = "/tmp/sparkinfer_lmcache_bench.sock";
    unlink(socket_path.c_str());
    const pid_t sidecar_pid = spawn_sidecar(python_env, script_env, socket_path, cfg, kv);
    if (sidecar_pid <= 0) {
        printf("could not spawn sidecar\n");
        return 1;
    }
    printf("[lmcache_bench] sidecar spawned (pid=%d), waiting for cold engine construction ...\n",
           (int)sidecar_pid);
    for (int i = 0; i < 300 && access(socket_path.c_str(), F_OK) != 0; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (access(socket_path.c_str(), F_OK) != 0) {
        printf("sidecar never came up within 30s\n");
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    sparkinfer::BridgeKVLayout layout;
    layout.num_layers = cfg.n_layers;
    layout.num_kv_heads = cfg.n_kv_heads;
    layout.head_dim = cfg.head_dim;
    layout.block_size = kv.block_size();
    layout.int8_kv = kv.int8_kv();
    layout.elem_bytes = kv.int8_kv() ? 1 : 2;
    layout.model_name = "lmcache-bench-model";
    sparkinfer::BridgeClient bridge(socket_path, layout);
    model.set_lmcache_bridge(&bridge);

    sparkinfer::ContinuousBatchEngine batch(&model, &kv, prompt_len + 64);

    std::vector<int> prompt;
    prompt.reserve(prompt_len);
    for (int i = 0; i < prompt_len; i++) prompt.push_back(1 + (i * 37) % (cfg.vocab - 1));

    // Request 1: cold, populates the cache (MISS). Not itself part of the comparison -- this is
    // the one-time cost of warming the cache, same as the plan's "session-close eviction"
    // trigger firing for the first time.
    sparkinfer::ContinuousBatchEngine::Request warm;
    warm.prompt = prompt;
    warm.max_new_tokens = 4;
    auto rw = batch.complete(warm);
    if (!rw.error.empty()) {
        printf("warm-up request failed: %s\n", rw.error.c_str());
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    printf("[lmcache_bench] warm-up (cache miss) TTFT: %.2f ms\n", rw.ttft_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));  // let the async STORE land

    // Request 2: same prompt, LMCache attached -- should hit.
    sparkinfer::ContinuousBatchEngine::Request cached;
    cached.prompt = prompt;
    cached.max_new_tokens = 4;
    auto rc = batch.complete(cached);
    if (!rc.error.empty()) {
        printf("cached request failed: %s\n", rc.error.c_str());
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    const uint64_t hits_after_cached = bridge.lookup_hit_count();
    printf("[lmcache_bench] cache-hit TTFT: %.2f ms (lookup hits=%llu)\n", rc.ttft_ms,
           (unsigned long long)hits_after_cached);

    // Request 3: same prompt again, LMCache DETACHED -- forces full recompute, isolating what
    // the cache actually saves (rather than comparing against the warm-up request, which paid
    // for the LOOKUP miss + STORE stage cost the baseline never pays either).
    model.set_lmcache_bridge(nullptr);
    sparkinfer::ContinuousBatchEngine::Request baseline;
    baseline.prompt = prompt;
    baseline.max_new_tokens = 4;
    auto rb = batch.complete(baseline);
    if (!rb.error.empty()) {
        printf("baseline request failed: %s\n", rb.error.c_str());
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    printf("[lmcache_bench] baseline (no cache, full recompute) TTFT: %.2f ms\n", rb.ttft_ms);

    terminate_sidecar(sidecar_pid);
    unlink(socket_path.c_str());

    printf("\n=== summary (prompt_len=%d tokens) ===\n", prompt_len);
    printf("baseline (full recompute):   %8.2f ms\n", rb.ttft_ms);
    printf("LMCache (cache hit):         %8.2f ms\n", rc.ttft_ms);
    if (rc.ttft_ms > 0.0)
        printf("speedup:                      %8.2fx\n", rb.ttft_ms / rc.ttft_ms);
    if (hits_after_cached < 1)
        printf("WARNING: no LOOKUP hit was recorded -- the \"cache hit\" number above did not "
               "actually hit the cache, treat it as invalid\n");
    return 0;
}
