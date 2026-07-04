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

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[kv] %s: %s\n", what, cudaGetErrorString(e));
}
constexpr int kMaxSeqs = 256;
constexpr int kMaxBlocksPerSeq = 4096;   // 4096 * block_size tokens per sequence
}

struct KVCacheManager::Impl {
    KVCacheConfig cfg;
    int total_blocks = 0;
    size_t layer_stride = 0;         // elements per layer in each pool
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    int* d_block_tables = nullptr;   // [kMaxSeqs, kMaxBlocksPerSeq]
    std::vector<int> free_list;
    std::unordered_map<uint64_t, std::vector<int>> seq_blocks;
    std::unordered_map<uint64_t, int> seq_slot;   // seq_id -> row in d_block_tables
    std::vector<int> free_slots;
};

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg, size_t pool_bytes)
    : impl_(new Impl()) {
    impl_->cfg = cfg;
    const size_t elems_per_block = (size_t)cfg.block_size * cfg.num_kv_heads * cfg.head_dim;
    const size_t bytes_per_block = elems_per_block * sizeof(unsigned short); // bf16
    // pool_bytes covers K and V across all layers.
    const size_t denom = (size_t)cfg.num_layers * 2 * bytes_per_block;
    impl_->total_blocks = denom ? (int)(pool_bytes / denom) : 0;
    impl_->layer_stride = (size_t)impl_->total_blocks * elems_per_block;

    const size_t pool_elems = (size_t)cfg.num_layers * impl_->layer_stride;
    cu(cudaMalloc(&impl_->k_pool, pool_elems * sizeof(unsigned short)), "malloc k_pool");
    cu(cudaMalloc(&impl_->v_pool, pool_elems * sizeof(unsigned short)), "malloc v_pool");
    cu(cudaMalloc(&impl_->d_block_tables, (size_t)kMaxSeqs * kMaxBlocksPerSeq * sizeof(int)), "malloc tables");

    impl_->free_list.reserve(impl_->total_blocks);
    for (int i = impl_->total_blocks - 1; i >= 0; --i) impl_->free_list.push_back(i);
    for (int i = kMaxSeqs - 1; i >= 0; --i) impl_->free_slots.push_back(i);
}

KVCacheManager::~KVCacheManager() {
    cudaFree(impl_->k_pool); cudaFree(impl_->v_pool); cudaFree(impl_->d_block_tables);
}

bool KVCacheManager::allocate(uint64_t seq_id, int num_tokens) {
    const int want = (num_tokens + impl_->cfg.block_size - 1) / impl_->cfg.block_size;
    if (want > kMaxBlocksPerSeq) return false;

    auto& blocks = impl_->seq_blocks[seq_id];
    const int have = (int)blocks.size();
    const int need = want - have;
    if (need <= 0) return true;  // already sufficient (including exact fit)

    const bool new_seq = impl_->seq_slot.find(seq_id) == impl_->seq_slot.end();
    if (new_seq && impl_->free_slots.empty()) return false;
    if ((int)impl_->free_list.size() < need) return false;

    for (int i = 0; i < need; i++) { blocks.push_back(impl_->free_list.back()); impl_->free_list.pop_back(); }

    int slot;
    auto it = impl_->seq_slot.find(seq_id);
    if (it != impl_->seq_slot.end()) slot = it->second;
    else { slot = impl_->free_slots.back(); impl_->free_slots.pop_back(); impl_->seq_slot[seq_id] = slot; }

    cu(cudaMemcpy(impl_->d_block_tables + (size_t)slot * kMaxBlocksPerSeq, blocks.data(),
                  blocks.size() * sizeof(int), cudaMemcpyHostToDevice), "copy block table");
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
    return impl_->d_block_tables + (size_t)it->second * kMaxBlocksPerSeq;
}

void*  KVCacheManager::k_pool() const { return impl_->k_pool; }
void*  KVCacheManager::v_pool() const { return impl_->v_pool; }
size_t KVCacheManager::layer_stride_elems() const { return impl_->layer_stride; }
int    KVCacheManager::block_size() const { return impl_->cfg.block_size; }
int    KVCacheManager::max_blocks_per_seq() const { return kMaxBlocksPerSeq; }
int    KVCacheManager::num_free_blocks() const { return (int)impl_->free_list.size(); }
int    KVCacheManager::num_total_blocks() const { return impl_->total_blocks; }

} // namespace sparkinfer
