// Paged KV-cache manager.
//
// One flat device pool holds K and V for every layer:
//   k_pool: [num_layers, num_blocks, block_size, num_kv_heads, head_dim] (bf16)
// A free-list of block ids backs allocation; each sequence gets a row in a
// device block-table array mapping its logical blocks to physical block ids,
// shared across layers (paging is layer-independent; the layer offset is applied
// to the pool base, not the table).

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/device_health.h"
#include <algorithm>
#include <atomic>

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    // Record context-killing errors so the engine can refuse new work instead of
    // issuing more against a dead context (see device_health.h).
    const bool fatal = note_cuda_error(e);
    // Rate-limit: a lost context makes EVERY subsequent call fail, which produced
    // 21,535 identical lines in one burst and buried the first, real error.
    static std::atomic<int> logged{0};
    const int n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n < 20 || fatal)
        fprintf(stderr, "[kv] %s: %s%s\n", what, cudaGetErrorString(e),
                fatal ? "  [CONTEXT LOST -- server will refuse further work]" : "");
    else if (n == 20)
        fprintf(stderr, "[kv] (further CUDA errors suppressed)\n");
}
constexpr int kMaxSeqs = 256;
}

struct KVCacheManager::Impl {
    KVCacheConfig cfg;
    int total_blocks = 0;
    int max_blocks_per_seq = 0;
    size_t layer_stride = 0;         // elements per layer in each pool
    size_t scale_layer_stride = 0;   // int8 path: fp16 scales per layer (= layer_stride / head_dim)
    bool int8_kv = false;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    int n_slots = 0;
    void* k_scale = nullptr;         // int8 path only: [num_layers, ..., 1] __half per (token, kv_head)
    void* v_scale = nullptr;
    int* d_block_tables = nullptr;   // [kMaxSeqs, max_blocks_per_seq]
    std::vector<int> free_list;
    std::unordered_map<uint64_t, std::vector<int>> seq_blocks;
    std::unordered_map<uint64_t, int> seq_slot;   // seq_id -> row in d_block_tables
    std::vector<int> free_slots;
};

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg, size_t pool_bytes)
    : impl_(new Impl()) {
    impl_->cfg = cfg;
    // int8 KV (Q8-style, per-token per-kv-head fp16 scale): 1 byte/elem + one scale per head vector,
    // halving the long-context KV read for the tensor-core flash-decode. Opt-in via cfg.int8_kv (the
    // Qwen3 example mains set it from SPARKINFER_KV_INT8, default on); other consumers stay bf16.
    impl_->int8_kv = cfg.int8_kv;
    const int elem_bytes = impl_->int8_kv ? 1 : (int)sizeof(unsigned short);
    const size_t elems_per_block = (size_t)cfg.block_size * cfg.num_kv_heads * cfg.head_dim;
    // total_blocks sized against the bf16 budget so callers/capacity are unchanged; int8 just mallocs
    // fewer bytes (+ the small scale pools).
    const size_t bytes_per_block = elems_per_block * sizeof(unsigned short); // bf16 budget
    // Only layers that actually hold KV get a pool slot (see KVCacheConfig::layer_slot).
    int n_slots = 0;
    for (int v : cfg.layer_slot) n_slots = (v + 1 > n_slots) ? v + 1 : n_slots;
    if (cfg.layer_slot.empty()) n_slots = cfg.num_layers;
    impl_->n_slots = n_slots;
    const size_t denom = (size_t)n_slots * 2 * bytes_per_block;
    impl_->total_blocks = denom ? (int)(pool_bytes / denom) : 0;
    // A sequence can never own more blocks than exist in the pool. Deriving the table stride
    // from that real limit avoids a separate 163,840-token ceiling that used to contradict the
    // server's advertised model context on large-context NVFP4 deployments.
    impl_->max_blocks_per_seq = std::max(1, impl_->total_blocks);
    impl_->layer_stride = (size_t)impl_->total_blocks * elems_per_block;

    // One extra TRAP slice when compacted. A layer with no slot has no KV by construction, so
    // reaching it means a caller ignored the layer typing; the trap absorbs that write instead of
    // letting it alias slot 0 (a real layer's KV) and corrupt attention silently. Costs one slice
    // of the n_slots the pool already shrank to.
    const int alloc_slices = n_slots + (cfg.layer_slot.empty() ? 0 : 1);
    const size_t pool_elems = (size_t)alloc_slices * impl_->layer_stride;
    cu(cudaMalloc(&impl_->k_pool, pool_elems * elem_bytes), "malloc k_pool");
    cu(cudaMalloc(&impl_->v_pool, pool_elems * elem_bytes), "malloc v_pool");
    if (impl_->int8_kv) {
        // one fp16 scale per (token slot, kv_head): scale stride = layer_stride / head_dim.
        impl_->scale_layer_stride = impl_->layer_stride / cfg.head_dim;
        const size_t scale_elems = (size_t)alloc_slices * impl_->scale_layer_stride;
        cu(cudaMalloc(&impl_->k_scale, scale_elems * sizeof(unsigned short)), "malloc k_scale");
        cu(cudaMalloc(&impl_->v_scale, scale_elems * sizeof(unsigned short)), "malloc v_scale");
    }
    cu(cudaMalloc(&impl_->d_block_tables,
                  (size_t)kMaxSeqs * impl_->max_blocks_per_seq * sizeof(int)), "malloc tables");

    impl_->free_list.reserve(impl_->total_blocks);
    for (int i = impl_->total_blocks - 1; i >= 0; --i) impl_->free_list.push_back(i);
    for (int i = kMaxSeqs - 1; i >= 0; --i) impl_->free_slots.push_back(i);
}

KVCacheManager::~KVCacheManager() {
    cudaFree(impl_->k_pool); cudaFree(impl_->v_pool); cudaFree(impl_->d_block_tables);
    if (impl_->k_scale) cudaFree(impl_->k_scale);
    if (impl_->v_scale) cudaFree(impl_->v_scale);
}

bool KVCacheManager::allocate(uint64_t seq_id, int num_tokens) {
    const int need = (num_tokens + impl_->cfg.block_size - 1) / impl_->cfg.block_size;
    if (need > impl_->max_blocks_per_seq) return false;

    auto& blocks = impl_->seq_blocks[seq_id];
    const int have = (int)blocks.size();
    if (have >= need) {
        auto it = impl_->seq_slot.find(seq_id);
        return it != impl_->seq_slot.end();
    }
    const int grow = need - have;
    if ((int)impl_->free_list.size() < grow) return false;

    int slot;
    auto it = impl_->seq_slot.find(seq_id);
    if (it != impl_->seq_slot.end()) slot = it->second;
    else {
        if (impl_->free_slots.empty()) return false;
        slot = impl_->free_slots.back();
        impl_->free_slots.pop_back();
        impl_->seq_slot[seq_id] = slot;
    }

    for (int i = 0; i < grow; i++) {
        blocks.push_back(impl_->free_list.back());
        impl_->free_list.pop_back();
    }

    cu(cudaMemcpy(impl_->d_block_tables + (size_t)slot * impl_->max_blocks_per_seq, blocks.data(),
                  blocks.size() * sizeof(int), cudaMemcpyHostToDevice), "copy block table");
    return true;
}

int KVCacheManager::allocated_tokens(uint64_t seq_id) const {
    auto it = impl_->seq_blocks.find(seq_id);
    if (it == impl_->seq_blocks.end()) return 0;
    return (int)it->second.size() * impl_->cfg.block_size;
}

int KVCacheManager::num_blocks(uint64_t seq_id) const {
    auto it = impl_->seq_blocks.find(seq_id);
    return it == impl_->seq_blocks.end() ? 0 : (int)it->second.size();
}

bool KVCacheManager::truncate_blocks(uint64_t seq_id, int keep_blocks) {
    if (keep_blocks < 0) return false;
    auto it = impl_->seq_blocks.find(seq_id);
    if (it == impl_->seq_blocks.end()) return false;
    auto& blocks = it->second;
    if ((int)blocks.size() <= keep_blocks) return true;
    while ((int)blocks.size() > keep_blocks) {
        impl_->free_list.push_back(blocks.back());
        blocks.pop_back();
    }
    auto sit = impl_->seq_slot.find(seq_id);
    if (sit == impl_->seq_slot.end()) return false;
    cu(cudaMemcpy(impl_->d_block_tables + (size_t)sit->second * impl_->max_blocks_per_seq, blocks.data(),
                  blocks.size() * sizeof(int), cudaMemcpyHostToDevice), "truncate block table");
    return true;
}

void KVCacheManager::free(uint64_t seq_id) {
    auto it = impl_->seq_blocks.find(seq_id);
    if (it != impl_->seq_blocks.end()) {
        for (int b : it->second) impl_->free_list.push_back(b);
        impl_->seq_blocks.erase(it);
    }
    auto s = impl_->seq_slot.find(seq_id);
    if (s != impl_->seq_slot.end()) { impl_->free_slots.push_back(s->second); impl_->seq_slot.erase(s); }
}

int* KVCacheManager::block_table(uint64_t seq_id) const {
    auto it = impl_->seq_slot.find(seq_id);
    if (it == impl_->seq_slot.end()) return nullptr;
    return impl_->d_block_tables + (size_t)it->second * impl_->max_blocks_per_seq;
}

const std::vector<int>& KVCacheManager::physical_block_ids(uint64_t seq_id) const {
    static const std::vector<int> kEmpty;
    auto it = impl_->seq_blocks.find(seq_id);
    return it == impl_->seq_blocks.end() ? kEmpty : it->second;
}

void*  KVCacheManager::k_pool() const { return impl_->k_pool; }
void*  KVCacheManager::v_pool() const { return impl_->v_pool; }
size_t KVCacheManager::layer_stride_elems() const { return impl_->layer_stride; }
int KVCacheManager::kv_slots() const { return impl_->n_slots; }
namespace {
// A slot-less layer means the caller ignored the layer typing the pool was built from. Say so once
// -- the trap slice keeps it from corrupting a real layer, but it is still a bug worth seeing.
int kv_trap_slot(const std::vector<int>& m, int layer) {
    const int s = (layer >= 0 && layer < (int)m.size()) ? m[layer] : -1;
    if (s < 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[kv] layer %d has no KV slot (linear/GDN layer) -- routed to the trap "
                            "slice; this layer should not be touching the paged pool\n", layer);
        }
    }
    return s;
}
}  // namespace

size_t KVCacheManager::layer_base_elems(int layer) const {
    const auto& m = impl_->cfg.layer_slot;
    if (m.empty()) return (size_t)layer * impl_->layer_stride;
    const int s = kv_trap_slot(m, layer);
    return (size_t)(s < 0 ? impl_->n_slots : s) * impl_->layer_stride;
}
size_t KVCacheManager::scale_layer_base_elems(int layer) const {
    const auto& m = impl_->cfg.layer_slot;
    if (m.empty()) return (size_t)layer * impl_->scale_layer_stride;
    const int s = kv_trap_slot(m, layer);
    return (size_t)(s < 0 ? impl_->n_slots : s) * impl_->scale_layer_stride;
}
bool   KVCacheManager::int8_kv() const { return impl_->int8_kv; }
void*  KVCacheManager::k_scale_pool() const { return impl_->k_scale; }
void*  KVCacheManager::v_scale_pool() const { return impl_->v_scale; }
size_t KVCacheManager::scale_layer_stride_elems() const { return impl_->scale_layer_stride; }
int    KVCacheManager::block_size() const { return impl_->cfg.block_size; }
int    KVCacheManager::max_blocks_per_seq() const { return impl_->max_blocks_per_seq; }
int    KVCacheManager::num_free_blocks() const { return (int)impl_->free_list.size(); }
int    KVCacheManager::num_total_blocks() const { return impl_->total_blocks; }

} // namespace sparkinfer
