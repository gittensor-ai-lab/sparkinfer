#include "model_engine.hpp"

#include "sparkinfer/gguf.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/runtime.h"

#include "../../runtime/examples/qwen3_gguf_config.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>

namespace sparkinfer_server {

namespace {

bool prompt_starts_with(const std::vector<int>& prompt, const std::vector<int>& prefix) {
    if (prefix.empty() || prompt.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), prompt.begin());
}

}  // namespace

struct ModelEngine::Impl {
    std::string path;
    sparkinfer::Qwen35Config cfg{};
    std::unique_ptr<sparkinfer::Runtime> rt;
    std::unique_ptr<sparkinfer::KVCacheManager> kv;
    std::unique_ptr<sparkinfer::moe::MoEEngine> engine;
    std::unique_ptr<sparkinfer::Qwen35Model> model;
    std::vector<int> prefix_tokens;
    bool ready = false;
};

ModelEngine::ModelEngine() : impl_(std::make_unique<Impl>()) {}
ModelEngine::~ModelEngine() = default;

bool ModelEngine::load(const std::string& gguf_path, int max_seq) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->ready = false;
    impl_->model.reset();
    impl_->engine.reset();
    impl_->kv.reset();
    impl_->rt.reset();
    impl_->path.clear();

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        fprintf(stderr, "[sparkinfer-server] no CUDA device\n");
        return false;
    }

    sparkinfer::GGUF g;
    if (!g.open(gguf_path)) {
        fprintf(stderr, "[sparkinfer-server] cannot open %s\n", gguf_path.c_str());
        return false;
    }

    impl_->cfg = sparkinfer::Qwen35Config{};
    qwen3_config_from_gguf(g, impl_->cfg);
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
    // Match qwen3_gguf_bench / qwen3_gguf_generate: hybrid Qwen3.6 uses int8 KV at ctx>=4k
    // (halves KV read bandwidth — the 32k decode win on 5090/PRO 6000). Override via env.
    { const char* e = getenv("SPARKINFER_KV_INT8");
      kvc.int8_kv = e ? (e[0] != '0')
                      : (impl_->cfg.hybrid ? (impl_->cfg.max_seq >= 4096) : true); }
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

    fprintf(stderr, "[sparkinfer-server] loading GGUF ...\n");
    if (!impl_->model->load_gguf(gguf_path)) {
        fprintf(stderr, "[sparkinfer-server] load_gguf failed\n");
        return false;
    }

    impl_->path = gguf_path;
    impl_->ready = true;
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

void ModelEngine::set_prefix_tokens(const std::vector<int>& tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->prefix_tokens = tokens;
}

int ModelEngine::prefix_token_len() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)impl_->prefix_tokens.size();
}

const std::string& ModelEngine::last_error() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
}

std::vector<int> ModelEngine::complete(const std::vector<int>& prompt_ids, int max_new_tokens,
                                       const sparkinfer::SamplingConfig* sampling) {
    return complete_streaming(prompt_ids, max_new_tokens, nullptr, sampling);
}

std::vector<int> ModelEngine::complete_streaming(const std::vector<int>& prompt_ids,
                                                 int max_new_tokens,
                                                 const std::function<void(int)>& on_token,
                                                 const sparkinfer::SamplingConfig* sampling) {
    std::lock_guard<std::mutex> lock(mu_);
    last_error_.clear();
    if (!impl_->ready || !impl_->model) {
        last_error_ = "model not loaded";
        return {};
    }
    if (prompt_ids.empty()) {
        last_error_ = "empty prompt";
        return {};
    }
    if (max_new_tokens <= 0) {
        last_error_ = "max_new_tokens must be positive";
        return {};
    }
    if ((int)prompt_ids.size() + max_new_tokens > impl_->cfg.max_seq) {
        last_error_ = "prompt + max_tokens exceeds context limit (" +
                      std::to_string(impl_->cfg.max_seq) + ")";
        fprintf(stderr, "[sparkinfer-server] context overflow: prompt=%zu max_new=%d max_seq=%d\n",
                prompt_ids.size(), max_new_tokens, impl_->cfg.max_seq);
        return {};
    }

    // Warm a shared prefix with batched prefill when configured; generate() reuses it for the
    // matching leading tokens and only token-loops the suffix.
    if (!impl_->prefix_tokens.empty()) {
        if (prompt_starts_with(prompt_ids, impl_->prefix_tokens)) {
            if (!impl_->model->cache_prefix(impl_->prefix_tokens)) {
                last_error_ = "cache_prefix failed (KV alloc or batched prefill)";
                fprintf(stderr, "[sparkinfer-server] %s\n", last_error_.c_str());
                return {};
            }
        } else {
            impl_->model->clear_prefix_cache();
        }
    } else {
        impl_->model->clear_prefix_cache();
    }
    std::vector<int> out = impl_->model->generate(prompt_ids, max_new_tokens, nullptr, sampling);
    if (on_token) {
        for (int t : out) on_token(t);
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        last_error_ = std::string("cuda error after decode: ") + cudaGetErrorString(e);
        fprintf(stderr, "[sparkinfer-server] %s\n", last_error_.c_str());
        return {};
    }
    if (out.empty() && max_new_tokens > 0 && last_error_.empty())
        last_error_ = "generate returned no tokens (KV alloc failure?)";
    return out;
}

}  // namespace sparkinfer_server
