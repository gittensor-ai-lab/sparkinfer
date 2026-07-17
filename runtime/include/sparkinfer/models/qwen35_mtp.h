#pragma once

#include "sparkinfer/models/qwen_config.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"

#include <cstdint>
#include <vector>

namespace sparkinfer {
namespace mtp {

// Weights for the single NextN / MTP block (blk.N in GGUF).
struct Weights {
    const void* eh_proj = nullptr;
    const void* enorm = nullptr;
    const void* hnorm = nullptr;
    const void* shared_head_norm = nullptr;
    int eh_proj_type = 0;
    Qwen35LayerWeights layer;
    bool loaded = false;
};

// Runtime state: separate 1-layer KV + scratch for MTP draft steps.
struct State {
    Weights w;
    KVCacheManager* kv = nullptr;
    uint64_t seq_id = 1;
    cudaStream_t stream = nullptr;
    std::vector<void*> owned;

    int qdim = 0;
    int kvdim = 0;
    int vocab_trim = 0;  // 0 = full vocab; else FastMTP top-N

    void* x = nullptr;
    void* xn = nullptr;
    void* concat = nullptr;
    void* qraw = nullptr;
    void* q = nullptr;
    void* qgate = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* attn = nullptr;
    void* ao = nullptr;
    void* h = nullptr;
    void* hn = nullptr;
    void* gate = nullptr;
    void* up = nullptr;
    void* fused = nullptr;
    float* logits = nullptr;
    int* d_scalars = nullptr;
    int* d_out_id = nullptr;
    void* aq81 = nullptr;
    float* fa_m = nullptr;
    float* fa_l = nullptr;
    float* fa_acc = nullptr;
    int n_splits = 32;
    int* mf_ids = nullptr;
    float* mf_w = nullptr;
    float* mf_h = nullptr;
    float* mf_out = nullptr;

    int pos = 0;
};

bool load_weights(GGUF& g, int blk, Weights& out, std::vector<void*>& owned,
                  bool qattn, const Qwen35Config& cfg);

void init_state(State& s, const Qwen35Config& cfg, KVCacheManager* kv, cudaStream_t stream);

void reset(State& s);

// Returns argmax; optionally writes the MTP hidden (pre-lm_head) into out_hidden[H].
int forward_step(State& s, const Qwen35Config& cfg,
                 int token_id, const void* main_hidden,
                 const void* embed_table, const void* lm_head, int lm_head_type,
                 int position, void* out_hidden = nullptr);

// Copy last forward_step logits to host (vocab_out floats). Returns vocab_out used.
int copy_logits(State& s, const Qwen35Config& cfg, float* host_logits);

}  // namespace mtp
}  // namespace sparkinfer
