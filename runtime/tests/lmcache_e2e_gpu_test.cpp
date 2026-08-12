// Live end-to-end test: a real sidecar subprocess (embedding the real `lmcache` package) +
// ContinuousBatchEngine + a small synthetic model on real hardware. Unlike
// lmcache_bridge_client_cpu_test.cpp (BridgeClient's own protocol/timeout behavior against a
// fake peer) and bridge/tests/test_bridge_e2e.py (the sidecar's own protocol handling against a
// raw socket client standing in for BridgeClient), this test is the one place both halves of the
// bridge run together against the real model/KV-cache code path -- proving the whole feature
// actually works, not just its two halves independently.
//
// A synthetic random-weight model (same pattern as batch_engine_gpu_test.cpp) is deliberately
// used instead of a real GGUF: this test verifies the CACHING MECHANISM's correctness (does a
// stored-then-restored KV chunk reproduce identical generation), which needs a real KV cache and
// a real deterministic decode loop, not real model quality -- avoiding a multi-GB model download
// keeps this test fast and independent of any specific model being available on the box.
//
// Skipped (not failed) when the sidecar's prerequisites aren't configured, matching the
// no-CUDA-device skip convention elsewhere in this test suite: this needs Python + a real
// lmcache/torch install (~1.5GB, CPU-only build), which most environments running the rest of
// this test suite won't have.
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

// Mirrors model_engine.cpp's spawn_lmcache_sidecar -- deliberately not shared code (this test
// wants to exercise the same CLI-args contract sparkinfer_server actually uses, independent of
// whether that function itself has a bug).
pid_t spawn_sidecar(const std::string& python, const std::string& script,
                    const std::string& socket_path, const sparkinfer::Qwen35Config& cfg,
                    const sparkinfer::KVCacheManager& kv) {
    std::vector<std::string> args = {
        python, script,
        "--socket", socket_path,
        "--instance-id", "lmcache-e2e-gpu-test",
        "--num-layers", std::to_string(cfg.n_layers),
        "--num-kv-heads", std::to_string(cfg.n_kv_heads),
        "--head-dim", std::to_string(cfg.head_dim),
        "--block-size", std::to_string(kv.block_size()),
        "--elem-bytes", std::to_string(kv.int8_kv() ? 1 : 2),
        "--model-name", "lmcache-e2e-gpu-test-model",
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
        fprintf(stderr, "[lmcache_e2e_gpu_test] exec failed: %s\n", strerror(errno));
        _exit(127);
    }
    return pid;
}

void terminate_sidecar(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 100; i++) {  // up to 10s -- cold sidecar teardown includes an LMCache
                                     // engine destroy, give it real room before SIGKILL
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
        printf("[SKIP] no CUDA device\n");
        return 0;
    }
    const char* python_env = getenv("SPARKINFER_LMCACHE_PYTHON");
    const char* script_env = getenv("SPARKINFER_LMCACHE_BRIDGE_SCRIPT");
    if (!python_env || !python_env[0] || !script_env || !script_env[0]) {
        printf("[SKIP] SPARKINFER_LMCACHE_PYTHON / SPARKINFER_LMCACHE_BRIDGE_SCRIPT not set "
               "(needs a real lmcache+torch venv -- see bridge/README.md)\n");
        return 0;
    }

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::Qwen35Config cfg;
    cfg.vocab = 4000;
    cfg.hidden = 2048;
    cfg.n_layers = 2;
    cfg.n_q_heads = 16;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 128;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.n_shared = 1;
    cfg.moe_ffn = 64;
    cfg.max_seq = 1024;  // must comfortably exceed the 256+ token prompt below
    cfg.eos_id = -1;

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    sparkinfer::KVCacheManager kv(kvc, 512ull * 1024 * 1024);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    sparkinfer::Qwen35Weights w;
    fill_weights(w, cfg);
    model.set_weights(w);

    const std::string socket_path = "/tmp/sparkinfer_lmcache_e2e_gpu_test.sock";
    unlink(socket_path.c_str());
    const pid_t sidecar_pid = spawn_sidecar(python_env, script_env, socket_path, cfg, kv);
    if (sidecar_pid <= 0) {
        printf("[FAIL] could not spawn sidecar\n");
        return 1;
    }
    printf("[lmcache_e2e_gpu_test] sidecar spawned (pid=%d), waiting for it to come up "
           "(cold engine construction takes ~10s) ...\n", (int)sidecar_pid);
    for (int i = 0; i < 300 && access(socket_path.c_str(), F_OK) != 0; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (access(socket_path.c_str(), F_OK) != 0) {
        printf("[FAIL] sidecar never created its socket within 30s\n");
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));  // let listen() start accepting

    sparkinfer::BridgeKVLayout layout;
    layout.num_layers = cfg.n_layers;
    layout.num_kv_heads = cfg.n_kv_heads;
    layout.head_dim = cfg.head_dim;
    layout.block_size = kv.block_size();
    layout.int8_kv = kv.int8_kv();
    layout.elem_bytes = kv.int8_kv() ? 1 : 2;
    layout.model_name = "lmcache-e2e-gpu-test-model";
    sparkinfer::BridgeClient bridge(socket_path, layout);
    model.set_lmcache_bridge(&bridge);

    sparkinfer::ContinuousBatchEngine batch(&model, &kv, 512);

    // 300 tokens: >= one 256-token LMCache chunk, so the LOOKUP gate in ingest_prompt_range
    // actually fires (see lmcache_chunk_size_tokens() in qwen35.cpp).
    std::vector<int> prompt;
    prompt.reserve(300);
    for (int i = 0; i < 300; i++) prompt.push_back(1 + (i * 37) % (cfg.vocab - 1));

    sparkinfer::ContinuousBatchEngine::Request req1;
    req1.prompt = prompt;
    req1.max_new_tokens = 8;
    auto r1 = batch.complete(req1);
    if (!r1.error.empty()) {
        printf("[FAIL] request 1: %s\n", r1.error.c_str());
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    printf("[lmcache_e2e_gpu_test] request 1 done: %zu tokens, lookup hits=%llu misses=%llu\n",
           r1.tokens.size(), (unsigned long long)bridge.lookup_hit_count(),
           (unsigned long long)bridge.lookup_miss_count());
    if (bridge.lookup_miss_count() < 1) {
        printf("[FAIL] expected at least 1 LOOKUP miss on the first request (nothing cached yet)\n");
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    if (bridge.lookup_hit_count() != 0) {
        printf("[FAIL] expected 0 LOOKUP hits on the first request, got %llu\n",
               (unsigned long long)bridge.lookup_hit_count());
        terminate_sidecar(sidecar_pid);
        return 1;
    }

    // The STORE this session's close triggers is fire-and-forget (BridgeClient's background
    // thread, plus the sidecar's own store() call) -- give it real time to land before the
    // second request's LOOKUP would otherwise race it.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    sparkinfer::ContinuousBatchEngine::Request req2;
    req2.prompt = prompt;  // identical prompt -- should now hit the stored chunk
    req2.max_new_tokens = 8;
    auto r2 = batch.complete(req2);
    if (!r2.error.empty()) {
        printf("[FAIL] request 2: %s\n", r2.error.c_str());
        terminate_sidecar(sidecar_pid);
        return 1;
    }
    printf("[lmcache_e2e_gpu_test] request 2 done: %zu tokens, lookup hits=%llu misses=%llu\n",
           r2.tokens.size(), (unsigned long long)bridge.lookup_hit_count(),
           (unsigned long long)bridge.lookup_miss_count());
    if (bridge.lookup_hit_count() < 1) {
        printf("[FAIL] expected at least 1 LOOKUP hit on the second (repeated-prompt) request -- "
               "the stored chunk from request 1 should have been found\n");
        terminate_sidecar(sidecar_pid);
        return 1;
    }

    // The actual correctness bar: a cache hit must reproduce byte-identical generation.
    // Greedy/deterministic decoding throughout this codebase makes this a strict, meaningful
    // assertion, not a "numerically close" one -- any mismatch means the restored KV bytes are
    // wrong (a corrupted scale, a block miscopy, a chunk-boundary/offset bug), not just "close
    // enough."
    if (r1.tokens != r2.tokens) {
        printf("[FAIL] generation mismatch between cache-miss and cache-hit runs of the same "
               "prompt -- restored KV must reproduce identical output. run1=[");
        for (int t : r1.tokens) printf("%d ", t);
        printf("] run2=[");
        for (int t : r2.tokens) printf("%d ", t);
        printf("]\n");
        terminate_sidecar(sidecar_pid);
        return 1;
    }

    terminate_sidecar(sidecar_pid);
    unlink(socket_path.c_str());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("[PASS] lmcache_e2e_gpu_test: cache-miss and cache-hit runs byte-identical "
           "(%zu tokens), hits=%llu misses=%llu\n",
           r1.tokens.size(), (unsigned long long)bridge.lookup_hit_count(),
           (unsigned long long)bridge.lookup_miss_count());
    return 0;
}
