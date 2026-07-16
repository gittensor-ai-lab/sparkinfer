// Paged KV-cache manager.
//
// One flat device pool holds K and V for every layer:
//   k_pool: [num_layers, num_blocks, block_size, num_kv_heads, head_dim] (bf16)
// A free-list of block ids backs allocation; each sequence gets a row in a
// device block-table array mapping its logical blocks to physical block ids,
// shared across layers (paging is layer-independent; the layer offset is applied
// to the pool base, not the table).

#include "sparkinfer/kv_cache.h"

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[kv] %s: %s\n", what, cudaGetErrorString(e));
}
constexpr int kMaxSeqs = 256;
constexpr int kMaxBlocksPerSeq = 10240;  // 10240 * 16 = 163840 tokens (128k ctx + decode headroom)
}

struct KVCacheManager::Impl {
    KVCacheConfig cfg;
    int total_blocks = 0;
    size_t layer_stride = 0;         // elements per layer in each pool
    size_t scale_layer_stride = 0;   // int8 path: fp16 scales per layer (= layer_stride / head_dim)
    bool int8_kv = false;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;         // int8 path only: [num_layers, ..., 1] __half per (token, kv_head)
    void* v_scale = nullptr;
    int* d_block_tables = nullptr;   // [kMaxSeqs, kMaxBlocksPerSeq]
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
    const size_t denom = (size_t)cfg.num_layers * 2 * bytes_per_block;
    impl_->total_blocks = denom ? (int)(pool_bytes / denom) : 0;
    impl_->layer_stride = (size_t)impl_->total_blocks * elems_per_block;

    const size_t pool_elems = (size_t)cfg.num_layers * impl_->layer_stride;
    cu(cudaMalloc(&impl_->k_pool, pool_elems * elem_bytes), "malloc k_pool");
    cu(cudaMalloc(&impl_->v_pool, pool_elems * elem_bytes), "malloc v_pool");
    if (impl_->int8_kv) {
        // one fp16 scale per (token slot, kv_head): scale stride = layer_stride / head_dim.
        impl_->scale_layer_stride = impl_->layer_stride / cfg.head_dim;
        const size_t scale_elems = (size_t)cfg.num_layers * impl_->scale_layer_stride;
        cu(cudaMalloc(&impl_->k_scale, scale_elems * sizeof(unsigned short)), "malloc k_scale");
        cu(cudaMalloc(&impl_->v_scale, scale_elems * sizeof(unsigned short)), "malloc v_scale");
    }
    cu(cudaMalloc(&impl_->d_block_tables, (size_t)kMaxSeqs * kMaxBlocksPerSeq * sizeof(int)), "malloc tables");

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
    if (need > kMaxBlocksPerSeq) return false;

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

    cu(cudaMemcpy(impl_->d_block_tables + (size_t)slot * kMaxBlocksPerSeq, blocks.data(),
                  blocks.size() * sizeof(int), cudaMemcpyHostToDevice), "copy block table");
    return true;
}

int KVCacheManager::allocated_tokens(uint64_t seq_id) const {
    auto it = impl_->seq_blocks.find(seq_id);
    if (it == impl_->seq_blocks.end()) return 0;
    return (int)it->second.size() * impl_->cfg.block_size;
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
    return impl_->d_block_tables + (size_t)it->second * kMaxBlocksPerSeq;
}

void*  KVCacheManager::k_pool() const { return impl_->k_pool; }
void*  KVCacheManager::v_pool() const { return impl_->v_pool; }
size_t KVCacheManager::layer_stride_elems() const { return impl_->layer_stride; }
bool   KVCacheManager::int8_kv() const { return impl_->int8_kv; }
void*  KVCacheManager::k_scale_pool() const { return impl_->k_scale; }
void*  KVCacheManager::v_scale_pool() const { return impl_->v_scale; }
size_t KVCacheManager::scale_layer_stride_elems() const { return impl_->scale_layer_stride; }
int    KVCacheManager::block_size() const { return impl_->cfg.block_size; }
int    KVCacheManager::max_blocks_per_seq() const { return kMaxBlocksPerSeq; }
int    KVCacheManager::num_free_blocks() const { return (int)impl_->free_list.size(); }
int    KVCacheManager::num_total_blocks() const { return impl_->total_blocks; }

} // namespace sparkinfer
