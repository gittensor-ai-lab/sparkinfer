#include "model_engine.hpp"
#include "sparkinfer/device_health.h"

#include "sparkinfer/gguf.h"
#include "sparkinfer/inference_engine.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/lmcache_bridge_client.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/qwen_vision.h"
#include "sparkinfer/models/qwen_vision_hf_config.h"
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "sparkinfer/safetensors.h"
#include "image_input.hpp"
#include "video_input.hpp"
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
        "--num-layers", std::to_string(cfg.n_layers),
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

    // Vision tower, when the checkpoint ships one. Owned here and handed to the batch engine by
    // pointer, so it must outlive it -- and must be freed while the CUDA context is still alive,
    // which is why reset_vision() is called from ~Impl's BODY rather than left to member order.
    sparkinfer::QwenVisionConfig vcfg{};
    sparkinfer::QwenVisionWeights vweights{};
    bool vision_ready = false;

    void reset_vision() {
        if (!vision_ready) return;
        free_qwen_vision_weights(vweights);
        vweights = sparkinfer::QwenVisionWeights{};
        vision_ready = false;
    }

    // Explicit teardown order: destroy the BridgeClient first (joins its ping/store threads,
    // which may otherwise be mid-handshake against the sidecar) before killing the sidecar out
    // from under it -- tearing them down in the other order risks those threads observing a
    // closed socket mid-operation instead of a clean, already-stopped state.
    ~Impl() {
        reset_vision();
        lmcache_bridge.reset();
        terminate_lmcache_sidecar(lmcache_sidecar_pid);
    }
};

ModelEngine::ModelEngine() : impl_(std::make_unique<Impl>()) {}
ModelEngine::~ModelEngine() = default;

bool ModelEngine::load(const std::string& gguf_path, int max_seq) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->ready = false;
    impl_->reset_vision();
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
    // Only the full-attention layers get a pool slot. The Gated-DeltaNet layers of a hybrid model
    // carry a recurrent state and never read paged KV, so a slot for them is pure waste -- on
    // Qwen3.8-27B that is 16 slots of 64, i.e. the pool was 4x larger than the model can use.
    // Every example main (qwen3_gguf_bench, qwen3_gguf_prefill_check, ...) has always set this;
    // the server never did, so the process that actually serves traffic was the one paying for it.
    // Left empty for non-hybrid models, where hybrid_kv_layer_slots gives every layer a slot
    // anyway and the behaviour is unchanged.
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(impl_->cfg.n_layers, impl_->cfg.hybrid,
                                                       impl_->cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, impl_->cfg.n_layers);
    const size_t epb = (size_t)16 * impl_->cfg.n_kv_heads * impl_->cfg.head_dim;
    const size_t blocks = (size_t)impl_->cfg.max_seq / 16 + 8;
    // pool_bytes is a bf16-DENOMINATED BUDGET, not an allocation: KVCacheManager derives
    // total_blocks = pool_bytes / (n_slots * 2 * bf16_bytes_per_block) and then mallocs at the
    // real element width (int8 just mallocs less). So the `* 2` here is the bf16 element size and
    // is correct as written -- it must stay even when int8_kv is on, or capacity halves. What has
    // to match kvc.layer_slot is the SLOT COUNT: passing n_layers while the manager counts 16
    // slots would hand out 4x the blocks for the same memory rather than shrinking the pool.
    impl_->kv = std::make_unique<sparkinfer::KVCacheManager>(
        kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    // Reports the slot count actually used, and the resident bytes rather than the bf16 budget --
    // the old line multiplied by n_layers (all 64) and by 2 regardless of int8, so it overstated
    // a hybrid int8 pool by 8x and was the number anyone sizing a deployment would have read.
    fprintf(stderr, "[sparkinfer-server] kv_cache: int8=%d slots=%d/%d blocks=%zu resident=%.1f GiB\n",
            kvc.int8_kv ? 1 : 0, kvL, impl_->cfg.n_layers, blocks,
            (double)kvL * 2.0 * epb * (kvc.int8_kv ? 1.0 : 2.0) * blocks
                / (1024.0 * 1024.0 * 1024.0));

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
            layout.num_layers = impl_->cfg.n_layers;
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

    // Vision tower. Absence is NOT an error -- a text-only checkpoint has no vision_config and
    // has_vision() stays false -- but a tower that is present and fails to load is reported here
    // rather than left to surface as a confusing per-request failure much later.
    if (is_dir) {
        std::string verr;
        if (!qwen_vision_config_from_hf_json(gguf_path, impl_->vcfg, verr)) {
            fprintf(stderr, "[sparkinfer-server] vision config malformed: %s\n", verr.c_str());
        } else if (impl_->vcfg.present) {
            sparkinfer::SafeTensorsModel vst;
            if (!vst.open(gguf_path)) {
                fprintf(stderr, "[sparkinfer-server] vision: cannot open safetensors\n");
            } else if (!load_qwen_vision_weights(vst, impl_->vcfg, impl_->vweights, verr)) {
                fprintf(stderr, "[sparkinfer-server] vision tower load failed: %s\n", verr.c_str());
            } else {
                impl_->vision_ready = true;
                impl_->batch_engine->set_vision(&impl_->vweights, &impl_->vcfg);
                fprintf(stderr, "[sparkinfer-server] vision tower ready: %d blocks, out_hidden=%d\n",
                        impl_->vcfg.depth, impl_->vcfg.out_hidden);
            }
        }
    }

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

bool ModelEngine::device_healthy() const { return !sparkinfer::device_lost(); }

bool ModelEngine::is_stop_token(int token_id) const {
    if (!impl_ || !impl_->ready || token_id < 0) return false;
    return token_id == impl_->cfg.eos_id ||
           (impl_->cfg.eos_id2 >= 0 && token_id == impl_->cfg.eos_id2);
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

bool ModelEngine::has_vision() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->vision_ready;
}

bool ModelEngine::prepare_images(const std::vector<std::string>& urls, int image_token_id,
                                 std::vector<int>& prompt_ids, PreparedImages& out,
                                 std::string& err) const {
    out.src_images.clear();
    out.src_videos.clear();
    out.images.clear();
    out.positions.clear();
    if (urls.empty()) return true;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->vision_ready) {
            err = "this model has no vision tower; image input is not supported";
            return false;
        }
    }
    // vcfg is written only by load() and read-only afterwards, and the tower weights are not
    // touched here at all -- this does CPU preprocessing only, so it runs without the engine
    // mutex and without contending for the GPU.
    for (size_t i = 0; i < urls.size(); i++) {
        const std::string where = "image " + std::to_string(i);
        std::vector<unsigned char> bytes;
        if (!parse_image_url(urls[i], bytes, err)) { err = where + ": " + err; return false; }
        DecodedImage img;
        if (!decode_image(bytes.data(), bytes.size(), img, err)) { err = where + ": " + err; return false; }
        PreparedImages::Image pi;
        int gh = 0, gw = 0;
        auto pixels = std::make_shared<std::vector<float>>();
        if (!sparkinfer::qwen_vision_preprocess(img.rgb.data(), img.height, img.width, impl_->vcfg,
                                                *pixels, &gh, &gw, err)) {
            err = where + ": " + err;
            return false;
        }
        pi.pixels = std::move(pixels);
        pi.grid_h = gh;
        pi.grid_w = gw;
        out.src_images.push_back(std::move(pi));
    }
    // One placeholder per image goes in, the count each grid needs comes out. A mismatch here is
    // an error, never a best-effort expansion: guessing would hand the model a prompt whose image
    // span is quietly truncated or padded, which it will describe fluently either way.
    return reexpand_images(image_token_id, prompt_ids, out, err);
}

bool ModelEngine::prepare_vision(const std::vector<std::string>& image_urls,
                                 const std::vector<std::string>& video_urls,
                                 int image_token_id, int video_token_id,
                                 int vision_start_token_id, int vision_end_token_id,
                                 const VideoSampling& sampling,
                                 const std::function<std::vector<int>(const std::string&)>& tokenize,
                                 std::vector<int>& prompt_ids, PreparedImages& out,
                                 std::string& err) const {
    out.src_images.clear();
    out.src_videos.clear();
    out.images.clear();
    out.positions.clear();
    if (image_urls.empty() && video_urls.empty()) return true;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->vision_ready) {
            err = "this model has no vision tower; image and video input are not supported";
            return false;
        }
    }
    if (!video_urls.empty() && !tokenize) {
        err = "video input requires a tokenizer for its timestamp markers";
        return false;
    }

    // Everything below is CPU-only preprocessing -- no tower weights are touched -- so it runs
    // without the engine mutex and without contending for the GPU.
    for (size_t i = 0; i < image_urls.size(); i++) {
        const std::string where = "image " + std::to_string(i);
        std::vector<unsigned char> bytes;
        if (!parse_image_url(image_urls[i], bytes, err)) { err = where + ": " + err; return false; }
        DecodedImage img;
        if (!decode_image(bytes.data(), bytes.size(), img, err)) { err = where + ": " + err; return false; }
        PreparedImages::Image pi;
        int gh = 0, gw = 0;
        auto pixels = std::make_shared<std::vector<float>>();
        if (!sparkinfer::qwen_vision_preprocess(img.rgb.data(), img.height, img.width, impl_->vcfg,
                                                *pixels, &gh, &gw, err)) {
            err = where + ": " + err;
            return false;
        }
        pi.pixels = std::move(pixels);
        pi.grid_h = gh;
        pi.grid_w = gw;
        out.src_images.push_back(std::move(pi));
    }

    for (size_t i = 0; i < video_urls.size(); i++) {
        const std::string where = "video " + std::to_string(i);
        if (!video_decoder_available(nullptr)) {
            err = where + ": video input needs ffmpeg and ffprobe on PATH";
            return false;
        }
        std::vector<unsigned char> bytes;
        if (!parse_video_url(video_urls[i], bytes, err)) { err = where + ": " + err; return false; }
        DecodedVideo clip;
        const int max_frames = sampling.max_frames > 0 ? sampling.max_frames : kDefaultMaxVideoFrames;
        if (!decode_video(bytes.data(), bytes.size(), max_frames, sampling.fps, clip, err)) {
            err = where + ": " + err;
            return false;
        }

        std::vector<const unsigned char*> frame_ptrs;
        frame_ptrs.reserve(clip.frames.size());
        for (const auto& f : clip.frames) frame_ptrs.push_back(f.data());

        std::vector<float> pixels;
        int gt = 0, gh = 0, gw = 0;
        if (!sparkinfer::qwen_vision_preprocess_video(frame_ptrs.data(), (int)frame_ptrs.size(),
                                                      clip.height, clip.width, impl_->vcfg,
                                                      pixels, &gt, &gh, &gw, err)) {
            err = where + ": " + err;
            return false;
        }

        PreparedImages::Video pv;
        pv.tokens_per_frame = sparkinfer::qwen_vision_num_tokens(gh, gw, impl_->vcfg);
        if (pv.tokens_per_frame <= 0) {
            err = where + ": video grid does not divide into the spatial merge block";
            return false;
        }

        // Slice the clip's one contiguous buffer into per-group buffers. Each group is an
        // ordinary tower call, and Image owns its pixels, so the copy buys a uniform contract
        // with the image path rather than a second offset-aware code path in the tower.
        const size_t per_group = pixels.size() / (size_t)(gt > 0 ? gt : 1);
        pv.groups.reserve((size_t)gt);
        for (int g = 0; g < gt; g++) {
            PreparedImages::Image gi;
            auto buf = std::make_shared<std::vector<float>>(
                pixels.begin() + (size_t)g * per_group,
                pixels.begin() + (size_t)(g + 1) * per_group);
            gi.pixels = std::move(buf);
            gi.grid_h = gh;
            gi.grid_w = gw;
            pv.groups.push_back(std::move(gi));
        }

        const std::vector<float> ts = sparkinfer::qwen_vision_video_timestamps(
            clip.frame_indices, clip.fps, impl_->vcfg.temporal_patch);
        pv.timestamp_tokens.reserve(pv.groups.size());
        for (size_t g = 0; g < pv.groups.size(); g++) {
            // Reference format: one decimal, e.g. "<1.5 seconds>". snprintf rather than
            // std::to_string, which is locale-independent here but fixed at six decimals.
            char buf[64];
            const float t = g < ts.size() ? ts[g] : 0.0f;
            std::snprintf(buf, sizeof(buf), "<%.1f seconds>", (double)t);
            pv.timestamp_tokens.push_back(tokenize(buf));
        }
        out.src_videos.push_back(std::move(pv));
    }

    return reexpand_vision(image_token_id, video_token_id,
                           vision_start_token_id, vision_end_token_id, prompt_ids, out, err);
}

bool ModelEngine::reexpand_images(int image_token_id, std::vector<int>& prompt_ids,
                                  PreparedImages& io, std::string& err) const {
    return reexpand_vision(image_token_id, impl_->vcfg.video_token_id,
                           impl_->vcfg.vision_start_token_id, impl_->vcfg.vision_end_token_id,
                           prompt_ids, io, err);
}

bool ModelEngine::reexpand_vision(int image_token_id, int video_token_id,
                                  int vision_start_token_id, int vision_end_token_id,
                                  std::vector<int>& prompt_ids, PreparedImages& io,
                                  std::string& err) const {
    io.positions.clear();
    io.images.clear();
    if (io.src_images.empty() && io.src_videos.empty()) return true;

    // VIDEOS FIRST, then images. Each expansion scans for its own placeholder id and preserves
    // every other token, so the two are independent -- but doing videos first keeps the image
    // expansion working on a prompt whose video spans are already their final length, which is
    // what makes the single ordering walk below correct.
    if (!io.src_videos.empty()) {
        std::vector<sparkinfer::QwenVideoSpan> spans;
        spans.reserve(io.src_videos.size());
        for (const auto& v : io.src_videos) {
            sparkinfer::QwenVideoSpan sp;
            sp.tokens_per_frame = v.tokens_per_frame;
            sp.timestamp_tokens = v.timestamp_tokens;
            spans.push_back(std::move(sp));
        }
        std::vector<int> expanded;
        if (!sparkinfer::qwen_vision_expand_video_placeholders(
                prompt_ids, video_token_id, vision_start_token_id, vision_end_token_id,
                spans, expanded, err))
            return false;
        prompt_ids.swap(expanded);
    }
    if (!io.src_images.empty()) {
        std::vector<int> counts;
        counts.reserve(io.src_images.size());
        for (const auto& im : io.src_images)
            counts.push_back(sparkinfer::qwen_vision_num_tokens(im.grid_h, im.grid_w, impl_->vcfg));
        std::vector<int> expanded;
        if (!sparkinfer::qwen_vision_expand_placeholders(prompt_ids, image_token_id, counts,
                                                         expanded, err))
            return false;
        prompt_ids.swap(expanded);
    }

    // Walk the finished prompt once and emit tower units in the order their placeholders appear.
    // Each MAXIMAL run of one placeholder id is one unit: an image's run is its whole grid, and a
    // video group's run is bounded by the vision_end/timestamp tokens between groups. Ordering by
    // the walk -- rather than concatenating images-then-videos -- is what makes an interleaved
    // request correct, because the engine pairs images[i] with positions[i] by index alone.
    size_t next_image = 0, next_video = 0, next_group = 0;
    for (size_t i = 0; i < prompt_ids.size(); ) {
        const int id = prompt_ids[i];
        if (id != image_token_id && id != video_token_id) { i++; continue; }
        size_t j = i;
        while (j < prompt_ids.size() && prompt_ids[j] == id) {
            io.positions.push_back((int)j);
            j++;
        }
        if (id == image_token_id) {
            if (next_image >= io.src_images.size()) {
                err = "prompt carries more image spans than images were preprocessed";
                return false;
            }
            io.images.push_back(io.src_images[next_image++]);
        } else {
            if (next_video >= io.src_videos.size() ||
                next_group >= io.src_videos[next_video].groups.size()) {
                err = "prompt carries more video frame spans than frames were preprocessed";
                return false;
            }
            io.images.push_back(io.src_videos[next_video].groups[next_group++]);
            if (next_group == io.src_videos[next_video].groups.size()) {
                next_video++;
                next_group = 0;
            }
        }
        i = j;
    }
    if (next_image != io.src_images.size() || next_video != io.src_videos.size()) {
        err = "prompt carries fewer vision spans than were preprocessed";
        return false;
    }

    // MRoPE positions, computed from the SAME walk-ordered unit list so the spans cannot drift
    // out of step with the placeholder runs they describe.
    io.mrope_pos.clear();
    io.mrope_decode_offset = 0;
    if (impl_->cfg.mrope()) {
        std::vector<sparkinfer::QwenVisionSpanGrid> grids;
        grids.reserve(io.images.size());
        // grid_t is 1 per unit: an image is one temporal group, and the reference splits a video's
        // grid into one t=1 row PER GROUP, which is exactly how the prompt spans them too.
        for (const auto& im : io.images) grids.push_back({1, im.grid_h, im.grid_w});
        if (!sparkinfer::qwen_vision_mrope_positions(prompt_ids, image_token_id, video_token_id,
                                                     grids, impl_->vcfg.spatial_merge, 0,
                                                     io.mrope_pos, err))
            return false;
        // The last token's rotary position + 1 is where decode resumes; the difference from the
        // prompt length is the constant every later decode step applies.
        if (!prompt_ids.empty()) {
            const int last_t = io.mrope_pos[(prompt_ids.size() - 1) * 3];
            io.mrope_decode_offset = (last_t + 1) - (int)prompt_ids.size();
        }
    }
    return true;
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
                                                     on_token_logprob,
                                                 const std::vector<int>& forced_tokens,
                                                 const PreparedImages* images) {
    CompletionResult out;
    sparkinfer::ContinuousBatchEngine::Request req;
    req.prompt = prompt_ids;
    req.max_new_tokens = max_new_tokens;
    req.forced_tokens = forced_tokens;
    if (images && !images->mrope_pos.empty()) {
        // Carried whenever the checkpoint declares an mrope_section, independently of whether this
        // particular request has images: the positions describe every token, and a prompt with no
        // vision span still gets the (degenerate, all-axes-equal) values, which cost nothing and
        // keep one code path instead of two.
        req.mrope_pos = images->mrope_pos;
        req.mrope_decode_offset = images->mrope_decode_offset;
    }
    if (images && !images->positions.empty()) {
        req.vision_pos = images->positions;
        req.vision_images.reserve(images->images.size());
        for (const auto& im : images->images)
            req.vision_images.push_back({im.pixels, im.grid_h, im.grid_w});   // shares the buffer
    }
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
        if (!forced_tokens.empty()) {
            if ((int)forced_tokens.size() != max_new_tokens) {
                out.error = "forced_tokens size must equal max_new_tokens";
                return out;
            }
            for (int t : forced_tokens) {
                if (t < 0 || t >= impl_->cfg.vocab) {
                    out.error = "forced token id out of range: " + std::to_string(t);
                    return out;
                }
            }
        }
        if ((int)prompt_ids.size() + max_new_tokens > impl_->cfg.max_seq) {
            out.error = "prompt + max_tokens exceeds context limit (" +
                        std::to_string(impl_->cfg.max_seq) + ")";
            fprintf(stderr, "[sparkinfer-server] context overflow: prompt=%zu max_new=%d max_seq=%d\n",
                    prompt_ids.size(), max_new_tokens, impl_->cfg.max_seq);
            return out;
        }

        // Shared prefix KV (session 0) is only safe when no other request is in-flight.
        //
        // Never used for a teacher-forced score. Taking the prefix path prefills the shared
        // prefix on its own (batched prefill at M = prefix_len) and only the suffix per request,
        // versus one batched pass at M = prompt_len otherwise -- a different tile decomposition,
        // so a few ULP of difference in the KV the score is computed against. Which branch runs
        // depends on whether another request happened to be in flight (prefix_exclusive), so with
        // a prefix configured the same /v1/score call could return marginally different logprobs
        // depending on server load. A scoring endpoint exists to be compared against another copy
        // of itself; co-tenancy-dependent numerics defeat that, and the prefill it saves is not
        // worth it.
        const bool prefix_match = forced_tokens.empty() && !impl_->prefix_tokens.empty() &&
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
