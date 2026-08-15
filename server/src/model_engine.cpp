#include "model_engine.hpp"

#include "sparkinfer/gguf.h"
#include "sparkinfer/inference_engine.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/lmcache_bridge_client.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/runtime.h"

#include "../../runtime/examples/qwen3_gguf_config.h"
#include "../../runtime/examples/qwen38_hf_config.h"

#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

namespace sparkinfer_server {

namespace {

bool prompt_starts_with(const std::vector<int>& prompt, const std::vector<int>& prefix) {
    if (prefix.empty() || prompt.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), prompt.begin());
}

int batch_tokens_per_step() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_BATCH_TOKENS");
        v = e ? std::max(1, atoi(e)) : 64;
    }
    return v;
}

// LMCache bridge sidecar lifecycle (docs/lmcache_bridge_protocol.md). Off by default -- gated on
// SPARKINFER_LMCACHE_ENABLE=1 -- so this feature ships with zero risk to existing behavior until
// explicitly turned on. All the actual cache logic lives behind Qwen35Model's BridgeClient*
// (null unless attached); this section only owns the sidecar subprocess and the socket path.

bool lmcache_enabled() {
    const char* e = getenv("SPARKINFER_LMCACHE_ENABLE");
    return e && e[0] == '1';
}

std::string lmcache_socket_path() {
    const char* e = getenv("SPARKINFER_LMCACHE_SOCKET");
    return e ? e : "/tmp/sparkinfer_lmcache.sock";
}

// fork+exec the sidecar (python3 bridge/lmcache_bridge.py --socket ... --num-layers ... ...),
// passing the KV layout as CLI args -- see bridge/lmcache_bridge.py's BridgeServer docstring for
// why this is CLI args and not deferred to the first HELLO (engine construction there is ~10s of
// first-time import + setup cost that must happen before any connection is accepted). Returns
// the child pid, or -1 on failure (missing SPARKINFER_LMCACHE_BRIDGE_SCRIPT, fork()/exec()
// failure) -- callers must treat -1 as "sidecar unavailable," never a fatal server error, matching
// the protocol doc's degradation invariant.
pid_t spawn_lmcache_sidecar(const std::string& socket_path, const sparkinfer::Qwen35Config& cfg,
                            const sparkinfer::KVCacheManager& kv, const std::string& model_name) {
    const char* script = getenv("SPARKINFER_LMCACHE_BRIDGE_SCRIPT");
    if (!script || !script[0]) {
        fprintf(stderr, "[sparkinfer-server] SPARKINFER_LMCACHE_ENABLE=1 but "
                        "SPARKINFER_LMCACHE_BRIDGE_SCRIPT is unset -- lmcache disabled\n");
        return -1;
    }
    const char* python_env = getenv("SPARKINFER_LMCACHE_PYTHON");
    const std::string python = python_env && python_env[0] ? python_env : "python3";
    const int elem_bytes = kv.int8_kv() ? 1 : 2;

    std::vector<std::string> args = {
        python,
        script,
        "--socket", socket_path,
        "--instance-id", "sparkinfer",
        // Sub-pools actually allocated (compacted for hybrids), matching BridgeKVLayout below.
        "--num-layers", std::to_string(kv.kv_layer_count()),
        "--num-kv-heads", std::to_string(cfg.n_kv_heads),
        "--head-dim", std::to_string(cfg.head_dim),
        "--block-size", std::to_string(kv.block_size()),
        "--elem-bytes", std::to_string(elem_bytes),
        "--model-name", model_name,
    };
    if (kv.int8_kv()) args.push_back("--int8-kv");

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[sparkinfer-server] lmcache sidecar fork() failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Child: stdout/stderr inherited from the parent (both point at the same stderr the
        // server itself logs to, per the doc -- no extra log plumbing needed to see sidecar
        // output). This spawn happens before sparkinfer_server ever binds its HTTP listener
        // (load() runs ahead of svr.listen() in main()), so there's no listening socket fd to
        // worry about leaking into the child; the CUDA context/device fds already open by this
        // point (weights are already loaded onto the GPU) are never touched by the child since
        // execvp runs immediately below with no CUDA calls in between, and NVIDIA's driver has
        // set O_CLOEXEC on its own device fds for years specifically to make this pattern safe.
        // execvp only returns on failure.
        execvp(python.c_str(), argv.data());
        fprintf(stderr, "[sparkinfer-server] lmcache sidecar exec failed: %s\n", strerror(errno));
        _exit(127);
    }
    fprintf(stderr, "[sparkinfer-server] lmcache sidecar spawned (pid=%d, socket=%s)\n",
            (int)pid, socket_path.c_str());
    return pid;
}

// SIGTERM + bounded wait (5s) before SIGKILL -- mirrors the HTTP listener's own graceful-
// shutdown grace period so a wedged sidecar never blocks server exit indefinitely.
void terminate_lmcache_sidecar(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) return;
        usleep(100000);
    }
    fprintf(stderr, "[sparkinfer-server] lmcache sidecar did not exit within 5s, sending SIGKILL\n");
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}

}  // namespace

struct ModelEngine::Impl {
    std::string path;
    sparkinfer::Qwen35Config cfg{};
    std::unique_ptr<sparkinfer::Runtime> rt;
    std::unique_ptr<sparkinfer::KVCacheManager> kv;
    std::unique_ptr<sparkinfer::moe::MoEEngine> engine;
    std::unique_ptr<sparkinfer::Qwen35Model> model;
    std::unique_ptr<sparkinfer::ContinuousBatchEngine> batch_engine;
    std::vector<int> prefix_tokens;
    bool ready = false;

    // LMCache bridge (docs/lmcache_bridge_protocol.md): the C++ socket client is owned here
    // (BridgeClient itself is declared in lmcache_bridge_client.h; Qwen35Model only holds a
    // non-owning pointer to it via set_lmcache_bridge()) alongside the sidecar subprocess this
    // ModelEngine spawned for it. Both null/-1 when the feature is disabled or unavailable.
    std::unique_ptr<sparkinfer::BridgeClient> lmcache_bridge;
    pid_t lmcache_sidecar_pid = -1;

    // Explicit teardown order: destroy the BridgeClient first (joins its ping/store threads,
    // which may otherwise be mid-handshake against the sidecar) before killing the sidecar out
    // from under it -- tearing them down in the other order risks those threads observing a
    // closed socket mid-operation instead of a clean, already-stopped state.
    ~Impl() {
        lmcache_bridge.reset();
        terminate_lmcache_sidecar(lmcache_sidecar_pid);
    }
};

ModelEngine::ModelEngine() : impl_(std::make_unique<Impl>()) {}
ModelEngine::~ModelEngine() = default;

bool ModelEngine::load(const std::string& gguf_path, int max_seq) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->ready = false;
    impl_->batch_engine.reset();
    impl_->model.reset();
    impl_->engine.reset();
    impl_->kv.reset();
    impl_->rt.reset();
    impl_->path.clear();
    // A reload (load() called again on an already-loaded engine) must not leak the previous
    // sidecar process or leave a BridgeClient pointing at a model that no longer exists.
    impl_->lmcache_bridge.reset();
    terminate_lmcache_sidecar(impl_->lmcache_sidecar_pid);
    impl_->lmcache_sidecar_pid = -1;

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        fprintf(stderr, "[sparkinfer-server] no CUDA device\n");
        return false;
    }

    // Three-way dispatch on what `-m` actually points at: a .gguf file (every model shipped so
    // far), or a directory -- which is either a HuggingFace "compressed-tensors" mixed FP8/NVFP4
    // checkpoint (config.json has a quantization_config block, e.g. unsloth/Qwen3.8-27B-NVFP4) or
    // a plain (unquantized) safetensors checkpoint. Same "auto-detect from what's actually there"
    // pattern the GGUF path already uses (general.architecture sniffing) -- no new CLI flag.
    struct stat path_st{};
    const bool is_dir = stat(gguf_path.c_str(), &path_st) == 0 && S_ISDIR(path_st.st_mode);
    enum class LoadKind { Gguf, CompressedTensors, PlainSafetensors };
    LoadKind kind = LoadKind::Gguf;
    if (is_dir) {
        std::ifstream cf(gguf_path + "/config.json");
        if (!cf) {
            fprintf(stderr, "[sparkinfer-server] %s is a directory but has no config.json\n",
                    gguf_path.c_str());
            return false;
        }
        nlohmann::json root;
        try { cf >> root; } catch (const std::exception& e) {
            fprintf(stderr, "[sparkinfer-server] %s/config.json parse error: %s\n",
                    gguf_path.c_str(), e.what());
            return false;
        }
        kind = root.contains("quantization_config") ? LoadKind::CompressedTensors
                                                      : LoadKind::PlainSafetensors;
    }

    sparkinfer::GGUF g;
    impl_->cfg = sparkinfer::Qwen35Config{};
    if (kind == LoadKind::Gguf) {
        if (!g.open(gguf_path)) {
            fprintf(stderr, "[sparkinfer-server] cannot open %s\n", gguf_path.c_str());
            return false;
        }
        qwen3_config_from_gguf(g, impl_->cfg);
    } else {
        // Both directory kinds share the same base HF config.json shape (text_config block) --
        // quantization_config only changes how load() below reads the actual weight bytes, not
        // the architecture/hyperparameter population.
        std::string err;
        if (!qwen38_config_from_hf_json(gguf_path, impl_->cfg, err)) {
            fprintf(stderr, "[sparkinfer-server] %s: %s\n", gguf_path.c_str(), err.c_str());
            return false;
        }
    }
    if (max_seq > 0) impl_->cfg.max_seq = max_seq;
    else if (impl_->cfg.max_seq < 2048) impl_->cfg.max_seq = 2048;

    fprintf(stderr, "[sparkinfer-server] arch %s, layers=%d, experts=%d top-%d, max_seq=%d\n",
            qwen3_model_label(impl_->cfg), impl_->cfg.n_layers, impl_->cfg.n_experts,
            impl_->cfg.top_k, impl_->cfg.max_seq);

    impl_->rt = sparkinfer::Runtime::create({});
    impl_->rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = impl_->cfg.n_layers;
    kvc.num_kv_heads = impl_->cfg.n_kv_heads;
    kvc.head_dim = impl_->cfg.head_dim;
    kvc.block_size = 16;
    { const char* e = getenv("SPARKINFER_KV_INT8");
      // Muse Glimmer: int8 KV cache is a confirmed correctness bug, not a precision tradeoff --
      // incoherent output from the very first decode token (#779), root-caused to its per-layer
      // sliding-window/NoPE alternation + sandwich-norm activations not matching what the int8
      // quantize/dequantize kernels were tuned against (Qwen3.6, same cfg.hybrid=true, is
      // unaffected). The CLI tools never caught this because their short eval prompts (<4096
      // tokens) always fell under the bf16 threshold below; the server activates int8 off its
      // configured max_seq (there's no per-request length at KV-pool-init time), and the default
      // max_seq (4096) satisfies ">=4096" unconditionally, so every default-config Muse Glimmer
      // server silently served garbage. Default to bf16 until the kernel bug itself is fixed;
      // SPARKINFER_KV_INT8=1 still force-enables it for anyone debugging that fix.
      kvc.int8_kv = e ? (e[0] != '0')
                      : (impl_->cfg.muse_glimmer ? false
                         : impl_->cfg.hybrid ? (impl_->cfg.max_seq >= 4096) : true); }
    // Hybrid: only every full_attn_interval-th layer owns KV (the rest are Gated-DeltaNet and keep
    // their state in the recurrent/conv buffers), so the pool only needs that many sub-pools.
    // SPARKINFER_KV_COMPACT=0 restores the full-width pool.
    {
        const char* kc = getenv("SPARKINFER_KV_COMPACT");
        if (!(kc && kc[0] == '0') && impl_->cfg.hybrid && impl_->cfg.full_attn_interval > 1 &&
            impl_->cfg.n_layers % impl_->cfg.full_attn_interval == 0)
            kvc.attn_every = impl_->cfg.full_attn_interval;
    }
    const size_t epb = (size_t)16 * impl_->cfg.n_kv_heads * impl_->cfg.head_dim;
    const size_t blocks = (size_t)impl_->cfg.max_seq / 16 + 8;
    impl_->kv = std::make_unique<sparkinfer::KVCacheManager>(
        kvc, (size_t)impl_->cfg.n_layers * 2 * epb * 2 * blocks);

    fprintf(stderr, "[sparkinfer-server] kv_cache: int8=%d blocks=%zu pool_budget=%.1f GiB\n",
            kvc.int8_kv ? 1 : 0, blocks,
            (double)impl_->cfg.n_layers * 2.0 * epb * 2.0 * blocks / (1024.0 * 1024.0 * 1024.0));

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = impl_->cfg.n_experts;
    mc.top_k = impl_->cfg.top_k;
    mc.hidden_dim = impl_->cfg.hidden;
    mc.ffn_dim = impl_->cfg.moe_ffn;
    mc.num_layers = impl_->cfg.n_layers;
    impl_->engine = sparkinfer::moe::MoEEngine::create(mc);

    impl_->model = std::make_unique<sparkinfer::Qwen35Model>(
        impl_->cfg, impl_->kv.get(), impl_->engine.get());

    if (kind == LoadKind::Gguf) {
        fprintf(stderr, "[sparkinfer-server] loading GGUF ...\n");
        if (!impl_->model->load_gguf(gguf_path)) {
            fprintf(stderr, "[sparkinfer-server] load_gguf failed\n");
            return false;
        }
    } else if (kind == LoadKind::CompressedTensors) {
        fprintf(stderr, "[sparkinfer-server] loading compressed-tensors checkpoint ...\n");
        if (!impl_->model->load_compressed_tensors(gguf_path)) {
            fprintf(stderr, "[sparkinfer-server] load_compressed_tensors failed\n");
            return false;
        }
    } else {
        // Qwen35Model::load_weights() reads a directory of already-converted flat .bin files
        // (runtime/tools/convert_qwen35.py's own offline output format), not a safetensors
        // directory directly -- no C++ loader for plain (unquantized) safetensors exists yet, this
        // config-detection branch was written ahead of that loader rather than left undetectable.
        fprintf(stderr, "[sparkinfer-server] %s: plain safetensors directories are not yet "
                        "supported directly -- convert with runtime/tools/convert_qwen35.py first "
                        "and pass the .bin output directory instead\n", gguf_path.c_str());
        return false;
    }

    if (lmcache_enabled()) {
        const std::string socket_path = lmcache_socket_path();
        impl_->lmcache_sidecar_pid =
            spawn_lmcache_sidecar(socket_path, impl_->cfg, *impl_->kv, gguf_path);
        if (impl_->lmcache_sidecar_pid > 0) {
            sparkinfer::BridgeKVLayout layout;
            // Sub-pools actually allocated (compacted for hybrids), not model layers.
            layout.num_layers = impl_->kv->kv_layer_count();
            layout.num_kv_heads = impl_->cfg.n_kv_heads;
            layout.head_dim = impl_->cfg.head_dim;
            layout.block_size = impl_->kv->block_size();
            layout.int8_kv = impl_->kv->int8_kv();
            layout.elem_bytes = impl_->kv->int8_kv() ? 1 : 2;
            layout.model_name = gguf_path;
            // Constructing BridgeClient does not block on the sidecar being ready yet (it
            // connects/handshakes lazily on first use, respecting its own timeout budget) --
            // safe to do immediately after fork() even though the sidecar's own ~10s cold-start
            // engine build is still running in the background.
            impl_->lmcache_bridge =
                std::make_unique<sparkinfer::BridgeClient>(socket_path, layout);
            impl_->model->set_lmcache_bridge(impl_->lmcache_bridge.get());
        }
        // lmcache_sidecar_pid <= 0 (spawn failure, e.g. missing SPARKINFER_LMCACHE_BRIDGE_SCRIPT)
        // leaves lmcache_bridge null -- the model's every lookup/store call site treats that as
        // "no cache tier," never a load failure. lmcache_enabled() being on with a broken sidecar
        // config is a misconfiguration worth the stderr line above, not a reason to refuse to serve.
    }

    sparkinfer::SchedulePolicy policy = sparkinfer::SchedulePolicy::CONTINUOUS_BATCHING;
    if (const char* p = getenv("SPARKINFER_SCHED_POLICY")) {
        // "chunked" / "chunked_prefill" → CHUNKED_PREFILL; "priority" → PRIORITY;
        // "continuous" (default) → CONTINUOUS_BATCHING. Match full keywords so
        // "continuous" is not misread as chunked (both start with 'c').
        if (strncmp(p, "chunk", 5) == 0) policy = sparkinfer::SchedulePolicy::CHUNKED_PREFILL;
        else if (p[0] == 'p' || p[0] == 'P') policy = sparkinfer::SchedulePolicy::PRIORITY;
        else policy = sparkinfer::SchedulePolicy::CONTINUOUS_BATCHING;
    }
    impl_->batch_engine = std::make_unique<sparkinfer::ContinuousBatchEngine>(
        impl_->model.get(), impl_->kv.get(), batch_tokens_per_step(), policy);

    impl_->path = gguf_path;
    impl_->ready = true;
    fprintf(stderr, "[sparkinfer-server] continuous batching enabled (policy=%d, batch=%d)\n",
            (int)policy, batch_tokens_per_step());
    fprintf(stderr, "[sparkinfer-server] model ready: %s\n", gguf_path.c_str());
    return true;
}

bool ModelEngine::loaded() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready;
}

std::string ModelEngine::model_path() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->path;
}

int ModelEngine::eos_id() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready ? impl_->cfg.eos_id : -1;
}

int ModelEngine::vocab() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready ? impl_->cfg.vocab : 0;
}

int ModelEngine::max_seq() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready ? impl_->cfg.max_seq : 0;
}

bool ModelEngine::is_museglimmer() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready && impl_->cfg.muse_glimmer;
}

bool ModelEngine::is_qwen38() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready && impl_->cfg.qwen38;
}

void ModelEngine::set_prefix_tokens(const std::vector<int>& tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->prefix_tokens = tokens;
}

int ModelEngine::prefix_token_len() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)impl_->prefix_tokens.size();
}

CompletionResult ModelEngine::complete(const std::vector<int>& prompt_ids, int max_new_tokens) {
    return complete_streaming(prompt_ids, max_new_tokens, nullptr);
}

int ModelEngine::active_requests() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->num_active() : 0;
}

int ModelEngine::free_kv_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->num_free_kv_blocks() : 0;
}

int ModelEngine::max_queue_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->max_queue_depth() : 0;
}

ModelEngine::LMCacheStats ModelEngine::lmcache_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    LMCacheStats stats;
    if (!impl_->lmcache_bridge) return stats;  // enabled=false, both counts 0
    stats.enabled = true;
    stats.lookup_hits = impl_->lmcache_bridge->lookup_hit_count();
    stats.lookup_misses = impl_->lmcache_bridge->lookup_miss_count();
    return stats;
}

CompletionResult ModelEngine::complete_streaming(const std::vector<int>& prompt_ids,
                                                 int max_new_tokens,
                                                 const std::function<bool(int)>& on_token,
                                                 float temperature, uint64_t seed,
                                                 int top_k, float top_p,
                                                 float presence_penalty, float frequency_penalty,
                                                 const std::vector<std::pair<int, float>>& logit_bias,
                                                 bool logprobs, int top_logprobs,
                                                 const std::function<void(const TokenLogprob&)>&
                                                     on_token_logprob) {
    CompletionResult out;
    sparkinfer::ContinuousBatchEngine::Request req;
    req.prompt = prompt_ids;
    req.max_new_tokens = max_new_tokens;
    req.temperature = temperature;
    req.seed = seed;
    req.top_k = top_k;
    req.top_p = top_p;
    req.presence_penalty = presence_penalty;
    req.frequency_penalty = frequency_penalty;
    req.logit_bias = logit_bias;
    req.logprobs = logprobs;
    req.top_logprobs = top_logprobs;

    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->ready || !impl_->model || !impl_->batch_engine) {
            out.error = "model not loaded";
            return out;
        }
        if (prompt_ids.empty()) {
            out.error = "empty prompt";
            return out;
        }
        if (max_new_tokens <= 0) {
            out.error = "max_new_tokens must be positive";
            return out;
        }
        if ((int)prompt_ids.size() + max_new_tokens > impl_->cfg.max_seq) {
            out.error = "prompt + max_tokens exceeds context limit (" +
                        std::to_string(impl_->cfg.max_seq) + ")";
            fprintf(stderr, "[sparkinfer-server] context overflow: prompt=%zu max_new=%d max_seq=%d\n",
                    prompt_ids.size(), max_new_tokens, impl_->cfg.max_seq);
            return out;
        }

        // Shared prefix KV (session 0) is only safe when no other request is in-flight.
        const bool prefix_match = !impl_->prefix_tokens.empty() &&
                                  prompt_starts_with(prompt_ids, impl_->prefix_tokens);
        const bool prefix_exclusive = impl_->batch_engine->num_active() == 0;
        if (prefix_match && prefix_exclusive) {
            if (impl_->model->prefix_cached_len() != (int)impl_->prefix_tokens.size()) {
                if (!impl_->model->cache_prefix(impl_->prefix_tokens)) {
                    out.error = "cache_prefix failed (KV alloc or batched prefill)";
                    fprintf(stderr, "[sparkinfer-server] %s\n", out.error.c_str());
                    return out;
                }
            }
            req.prefill_start = (int)impl_->prefix_tokens.size();
            req.use_prefix_session = true;
        } else {
            // clear_prefix_cache() frees whatever session is currently active on the shared
            // Qwen35Model (kv->free(active_seq_id)) -- but the continuous-batch worker thread
            // calls activate_session() for whichever job it's stepping right now, on its own
            // thread, independent of this mutex. Calling clear here without the same
            // prefix_exclusive guard the "use prefix" branch above already has meant an
            // unrelated new request (any request that doesn't match the prefix -- the common
            // case whenever no prefix is configured at all) could free the KV blocks out from
            // under an actively-decoding, completely unrelated job. Reproduced directly: under
            // concurrent load this corrupts the KV cache ("[kv] copy block table: an illegal
            // memory access was encountered", poisoning the CUDA context for the rest of the
            // process). Skipping the clear when non-exclusive just defers it -- cache_prefix()
            // already re-primes correctly the next time the prefix session is actually used.
            if (!prefix_match && prefix_exclusive) impl_->model->clear_prefix_cache();
            req.prefill_start = 0;
            req.use_prefix_session = false;
        }
    }

    // complete_streaming releases the engine mutex above so other HTTP workers can enqueue.
    // Errors must travel with this stack frame — a shared last_error_ slot would let one
    // request clear or observe another request's failure under concurrency.
    //
    // Pass nullptr straight through (not a lambda that internally no-ops) when the caller didn't
    // ask for logprobs -- ContinuousBatchEngine::step_job()'s cost gate is
    // `req.logprobs && on_token_logprob`, so an always-non-null glue lambda here would defeat the
    // "logprobs=false costs nothing extra" property this whole design depends on.
    std::function<void(const sparkinfer::Qwen35Model::TokenLogprob&)> glue_logprob;
    if (on_token_logprob) {
        glue_logprob = [&on_token_logprob](const sparkinfer::Qwen35Model::TokenLogprob& tl) {
            TokenLogprob mirrored;
            mirrored.token_id = tl.token_id;
            mirrored.logprob = tl.logprob;
            mirrored.top_alternatives = tl.top_alternatives;
            on_token_logprob(mirrored);
        };
    }
    auto result = impl_->batch_engine->complete_streaming(req, on_token, glue_logprob);

    std::lock_guard<std::mutex> lock(mu_);
    out.overloaded = result.overloaded;
    out.alloc_failed = result.alloc_failed;
    out.timed_out = result.timed_out;
    out.cancelled = result.cancelled;
    out.reached_token_limit = result.reached_token_limit;
    out.ttft_ms = result.ttft_ms;
    out.generation_ms = result.generation_ms;
    out.decode_tps = result.decode_tps;
    if (!result.error.empty()) {
        out.error = result.error;
        fprintf(stderr, "[sparkinfer-server] %s\n", out.error.c_str());
        return out;
    }
    if (result.cancelled) {
        out.tokens = std::move(result.tokens);
        return out;
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        out.error = std::string("cuda error after decode: ") + cudaGetErrorString(e);
        fprintf(stderr, "[sparkinfer-server] %s\n", out.error.c_str());
        return out;
    }
    if (result.tokens.empty() && max_new_tokens > 0)
        out.error = "generate returned no tokens (KV alloc failure?)";
    out.tokens = std::move(result.tokens);
    return out;
}

}  // namespace sparkinfer_server
