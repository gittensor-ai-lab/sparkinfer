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
#include <cstdint>

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
    size_t elems_per_block = 0;      // block_size * num_kv_heads * head_dim
    size_t layer_stride = 0;         // elements per layer in each pool
    size_t scale_layer_stride = 0;   // int8 path: fp16 scales per layer (= layer_stride / head_dim)
    bool int8_kv = false;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    int n_slots = 0;
    void* k_scale = nullptr;         // int8 path only: [num_layers, ..., 1] __half per (token, kv_head)
    void* v_scale = nullptr;
    int* d_block_tables = nullptr;   // [kMaxSeqs, max_blocks_per_seq]
    // Windowed slots (KVCacheConfig::window_tokens). ring_blocks is the per-sequence ring; the
    // windowed slices live after the full ones in the same pool, indexed by their own block ids
    // and their own free list. d_win_tables holds ring-repeated rows so the append/attention
    // kernels stay unchanged.
    bool win_on = false;
    int n_win_slots = 0;             // windowed slices
    int n_full_slots = 0;            // full-context slices
    int ring_blocks = 0;             // blocks in one sequence's ring
    int win_total_blocks = 0;        // windowed blocks in the pool (ring_blocks * sessions)
    int win_pass_tokens = 0;
    size_t win_base_elems = 0;       // element offset of the first windowed slice
    size_t win_scale_base_elems = 0;
    std::vector<int> slot_group_index;   // slot -> index within its group
    std::vector<char> slot_is_win;       // slot -> windowed?
    int* d_win_tables = nullptr;         // [kMaxSeqs, max_blocks_per_seq], ring-repeated
    std::vector<int> free_list_win;
    std::unordered_map<uint64_t, std::vector<int>> seq_ring;   // seq -> its ring blocks
    std::vector<int> free_list;
    // Holders per physical block: every sequence listing it plus every retained list (prefix
    // cache entries). A block is on free_list exactly when its count is zero.
    std::vector<int> refs;

    void unref(int b) {
        if (b < 0 || b >= (int)refs.size() || refs[b] <= 0) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                fprintf(stderr, "[kv] released block %d that has no holder -- refcount bug\n", b);
            return;
        }
        if (--refs[b] == 0) free_list.push_back(b);
    }
    // Take a ring for seq_id if it has none. The device row is written once, ring-repeated over
    // the whole table stride, so a later grow() never has to touch it again.
    bool ensure_ring(uint64_t seq_id, int slot) {
        if (!win_on) return true;
        auto it = seq_ring.find(seq_id);
        if (it != seq_ring.end() && (int)it->second.size() == ring_blocks) return true;
        if ((int)free_list_win.size() < ring_blocks) return false;
        std::vector<int>& ring = seq_ring[seq_id];
        ring.clear(); ring.reserve((size_t)ring_blocks);
        for (int i = 0; i < ring_blocks; ++i) {
            ring.push_back(free_list_win.back());
            free_list_win.pop_back();
        }
        std::vector<int> row((size_t)max_blocks_per_seq);
        for (int i = 0; i < max_blocks_per_seq; ++i) row[(size_t)i] = ring[(size_t)(i % ring_blocks)];
        cu(cudaMemcpy(d_win_tables + (size_t)slot * max_blocks_per_seq, row.data(),
                      row.size() * sizeof(int), cudaMemcpyHostToDevice), "copy ring table");
        return true;
    }
    void release_ring(uint64_t seq_id) {
        auto it = seq_ring.find(seq_id);
        if (it == seq_ring.end()) return;
        for (int b : it->second) free_list_win.push_back(b);
        seq_ring.erase(it);
    }
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
    impl_->elems_per_block = elems_per_block;

    // WINDOWED SLICES. Capping needs the window, the pass bound, the context the pool was sized
    // for, and a per-slot map with both kinds in it; anything missing keeps the old single-group
    // layout. SPARKINFER_KV_SWA_CAP=0 forces it off out of the same binary.
    const int blocks_per_seq_full = cfg.max_seq_tokens > 0
        ? (cfg.max_seq_tokens + cfg.block_size - 1) / cfg.block_size : 0;
    {
        const char* e = getenv("SPARKINFER_KV_SWA_CAP");
        const bool env_off = e && e[0] == '0';
        int n_win = 0;
        for (char c : cfg.slot_windowed) n_win += c ? 1 : 0;
        // The ring must hold the window AND one pass on top of it: a prefill pass appends every
        // token before its attention runs, so the pass's first query still reads window_tokens
        // back while the pass's last token has already been written. One spare block covers the
        // block-alignment of both ends.
        const long ring_tokens = (long)cfg.window_tokens + cfg.window_pass_tokens;
        const int ring_blocks = ring_tokens > 0
            ? (int)((ring_tokens + cfg.block_size - 1) / cfg.block_size) + 1 : 0;
        impl_->win_on = !env_off && cfg.window_tokens > 0 && cfg.window_pass_tokens > 0 &&
                        blocks_per_seq_full > 0 && n_win > 0 &&
                        n_win < n_slots && ring_blocks < blocks_per_seq_full &&
                        (int)cfg.slot_windowed.size() == n_slots;
        if (impl_->win_on) {
            impl_->ring_blocks = ring_blocks;
            impl_->win_pass_tokens = cfg.window_pass_tokens;
            impl_->n_win_slots = n_win;
            impl_->n_full_slots = n_slots - n_win;
            impl_->slot_is_win.assign(cfg.slot_windowed.begin(), cfg.slot_windowed.end());
            impl_->slot_group_index.assign((size_t)n_slots, 0);
            int nf = 0, nw = 0;
            for (int i = 0; i < n_slots; ++i)
                impl_->slot_group_index[(size_t)i] = impl_->slot_is_win[(size_t)i] ? nw++ : nf++;
            // Rings for as many sequences as the full group can hold at the sized context, so a
            // pool that can open N sessions can also give N of them a ring.
            const int sessions = std::max(1, impl_->total_blocks / blocks_per_seq_full);
            impl_->win_total_blocks = sessions * ring_blocks;
            impl_->free_list_win.reserve((size_t)impl_->win_total_blocks);
            for (int i = impl_->win_total_blocks - 1; i >= 0; --i) impl_->free_list_win.push_back(i);
            fprintf(stderr, "[kv] windowed slots: %d of %d slices capped at %d blocks "
                    "(%d tokens) vs %d, %d sessions -- pool %.2f GB instead of %.2f GB\n",
                    n_win, n_slots, ring_blocks, ring_blocks * cfg.block_size, blocks_per_seq_full,
                    sessions,
                    (double)((size_t)(impl_->n_full_slots * (size_t)impl_->total_blocks +
                              (size_t)n_win * impl_->win_total_blocks) * 2 * elems_per_block *
                             elem_bytes) / 1e9,
                    (double)((size_t)n_slots * impl_->total_blocks * 2 * elems_per_block *
                             elem_bytes) / 1e9);
        }
    }

    // One extra TRAP slice when compacted. A layer with no slot has no KV by construction, so
    // reaching it means a caller ignored the layer typing; the trap absorbs that write instead of
    // letting it alias slot 0 (a real layer's KV) and corrupt attention silently. Costs one slice
    // of the n_slots the pool already shrank to.
    const int alloc_slices = n_slots + (cfg.layer_slot.empty() ? 0 : 1);
    // Full slices first, then the (smaller) windowed ones, then the trap slice -- which sits in
    // the full-size region so a stray write to a slot-less layer still lands somewhere legal.
    const size_t win_stride = (size_t)impl_->win_total_blocks * elems_per_block;
    const size_t pool_elems = impl_->win_on
        ? ((size_t)(impl_->n_full_slots + (alloc_slices - n_slots)) * impl_->layer_stride +
           (size_t)impl_->n_win_slots * win_stride)
        : (size_t)alloc_slices * impl_->layer_stride;
    if (impl_->win_on) {
        // Windowed region starts after every full slice AND the trap slice.
        impl_->win_base_elems =
            (size_t)(impl_->n_full_slots + (alloc_slices - n_slots)) * impl_->layer_stride;
    }
    cu(cudaMalloc(&impl_->k_pool, pool_elems * elem_bytes), "malloc k_pool");
    cu(cudaMalloc(&impl_->v_pool, pool_elems * elem_bytes), "malloc v_pool");
    if (impl_->int8_kv) {
        // one fp16 scale per (token slot, kv_head): scale stride = layer_stride / head_dim.
        impl_->scale_layer_stride = impl_->layer_stride / cfg.head_dim;
        const size_t win_scale_stride = win_stride / cfg.head_dim;
        const size_t scale_elems = impl_->win_on
            ? ((size_t)(impl_->n_full_slots + (alloc_slices - n_slots)) * impl_->scale_layer_stride +
               (size_t)impl_->n_win_slots * win_scale_stride)
            : (size_t)alloc_slices * impl_->scale_layer_stride;
        if (impl_->win_on)
            impl_->win_scale_base_elems =
                (size_t)(impl_->n_full_slots + (alloc_slices - n_slots)) * impl_->scale_layer_stride;
        cu(cudaMalloc(&impl_->k_scale, scale_elems * sizeof(unsigned short)), "malloc k_scale");
        cu(cudaMalloc(&impl_->v_scale, scale_elems * sizeof(unsigned short)), "malloc v_scale");
    }
    cu(cudaMalloc(&impl_->d_block_tables,
                  (size_t)kMaxSeqs * impl_->max_blocks_per_seq * sizeof(int)), "malloc tables");
    if (impl_->win_on)
        cu(cudaMalloc(&impl_->d_win_tables,
                      (size_t)kMaxSeqs * impl_->max_blocks_per_seq * sizeof(int)), "malloc ring tables");

    impl_->free_list.reserve(impl_->total_blocks);
    for (int i = impl_->total_blocks - 1; i >= 0; --i) impl_->free_list.push_back(i);
    impl_->refs.assign(impl_->total_blocks, 0);
    for (int i = kMaxSeqs - 1; i >= 0; --i) impl_->free_slots.push_back(i);
}

KVCacheManager::~KVCacheManager() {
    cudaFree(impl_->k_pool); cudaFree(impl_->v_pool); cudaFree(impl_->d_block_tables);
    if (impl_->d_win_tables) cudaFree(impl_->d_win_tables);
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

    // The ring comes first: it is all-or-nothing per sequence, and a sequence that cannot get one
    // must not end up holding full blocks its windowed layers have nowhere to write.
    if (!impl_->ensure_ring(seq_id, slot)) {
        if (have == 0) {
            impl_->free_slots.push_back(slot);
            impl_->seq_slot.erase(seq_id);
        }
        return false;
    }
    for (int i = 0; i < grow; i++) {
        const int b = impl_->free_list.back();
        impl_->free_list.pop_back();
        impl_->refs[b] = 1;
        blocks.push_back(b);
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
        impl_->unref(blocks.back());
        blocks.pop_back();
    }
    auto sit = impl_->seq_slot.find(seq_id);
    if (sit == impl_->seq_slot.end()) return false;
    cu(cudaMemcpy(impl_->d_block_tables + (size_t)sit->second * impl_->max_blocks_per_seq, blocks.data(),
                  blocks.size() * sizeof(int), cudaMemcpyHostToDevice), "truncate block table");
    return true;
}

void KVCacheManager::free(uint64_t seq_id) {
    impl_->release_ring(seq_id);
    auto it = impl_->seq_blocks.find(seq_id);
    if (it != impl_->seq_blocks.end()) {
        for (int b : it->second) impl_->unref(b);
        impl_->seq_blocks.erase(it);
    }
    auto s = impl_->seq_slot.find(seq_id);
    if (s != impl_->seq_slot.end()) { impl_->free_slots.push_back(s->second); impl_->seq_slot.erase(s); }
}

std::vector<int> KVCacheManager::retain_prefix_blocks(uint64_t seq_id, int n_blocks) {
    std::vector<int> out;
    auto it = impl_->seq_blocks.find(seq_id);
    if (n_blocks <= 0 || it == impl_->seq_blocks.end() || (int)it->second.size() < n_blocks) return out;
    out.assign(it->second.begin(), it->second.begin() + n_blocks);
    for (int b : out) impl_->refs[b]++;
    return out;
}

void KVCacheManager::release_blocks(const std::vector<int>& physical_ids) {
    for (int b : physical_ids) impl_->unref(b);
}

bool KVCacheManager::allocate_with_prefix(uint64_t seq_id, const std::vector<int>& prefix, int num_tokens) {
    if (prefix.empty()) return allocate(seq_id, num_tokens);
    // A windowed slice is a ring PRIVATE to one sequence, so another sequence's prefix blocks do
    // not name its window -- sharing them would read a stranger's KV in 39 of 52 Muse layers.
    // Refuse, and say so once: callers ask prefix_sharing_supported() and recompute instead.
    if (impl_->win_on) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            fprintf(stderr, "[kv] prefix-block sharing refused: this pool has windowed (ring) "
                            "slices, which are per-sequence -- the caller must recompute the "
                            "prefix (SPARKINFER_KV_SWA_CAP=0 restores shareable full slices)\n");
        return false;
    }
    const int need = (num_tokens + impl_->cfg.block_size - 1) / impl_->cfg.block_size;
    if (need > impl_->max_blocks_per_seq || (int)prefix.size() > need) return false;
    auto existing = impl_->seq_blocks.find(seq_id);
    if (existing != impl_->seq_blocks.end() && !existing->second.empty()) return false;
    // A block nobody holds is back on the free list and may already belong to someone else.
    for (int b : prefix)
        if (b < 0 || b >= impl_->total_blocks || impl_->refs[b] <= 0) return false;
    const int grow = need - (int)prefix.size();
    if ((int)impl_->free_list.size() < grow) return false;

    int slot;
    auto sit = impl_->seq_slot.find(seq_id);
    if (sit != impl_->seq_slot.end()) slot = sit->second;
    else {
        if (impl_->free_slots.empty()) return false;
        slot = impl_->free_slots.back();
        impl_->free_slots.pop_back();
        impl_->seq_slot[seq_id] = slot;
    }

    auto& blocks = impl_->seq_blocks[seq_id];
    blocks = prefix;
    for (int b : prefix) impl_->refs[b]++;
    for (int i = 0; i < grow; i++) {
        const int b = impl_->free_list.back();
        impl_->free_list.pop_back();
        impl_->refs[b] = 1;
        blocks.push_back(b);
    }
    cu(cudaMemcpy(impl_->d_block_tables + (size_t)slot * impl_->max_blocks_per_seq, blocks.data(),
                  blocks.size() * sizeof(int), cudaMemcpyHostToDevice), "copy shared block table");
    return true;
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
int kv_trap_slot_impl(const std::vector<int>& m, int layer) {
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

namespace {
// Slot index for `layer`, or -1 for the trap slice. Identity map = every layer has its own slot.
inline int kv_slot_of(const std::vector<int>& m, int layer) {
    return m.empty() ? layer : kv_trap_slot_impl(m, layer);
}
}  // namespace
// Base of `layer`'s slice. With windowed slots the pool holds two differently sized groups --
// [full slices][trap][windowed slices] -- so this walks the layer's own group. It is the only
// place that knows the split: every caller already takes its per-layer pointer from here.
size_t KVCacheManager::layer_base_elems(int layer) const {
    const int s = kv_slot_of(impl_->cfg.layer_slot, layer);
    if (!impl_->win_on)
        return (size_t)(s < 0 ? impl_->n_slots : s) * impl_->layer_stride;
    if (s < 0) return (size_t)impl_->n_full_slots * impl_->layer_stride;   // trap slice
    const int gi = impl_->slot_group_index[(size_t)s];
    if (impl_->slot_is_win[(size_t)s])
        return impl_->win_base_elems +
               (size_t)gi * ((size_t)impl_->win_total_blocks * impl_->elems_per_block);
    return (size_t)gi * impl_->layer_stride;
}
size_t KVCacheManager::scale_layer_base_elems(int layer) const {
    const int s = kv_slot_of(impl_->cfg.layer_slot, layer);
    if (!impl_->win_on)
        return (size_t)(s < 0 ? impl_->n_slots : s) * impl_->scale_layer_stride;
    if (s < 0) return (size_t)impl_->n_full_slots * impl_->scale_layer_stride;
    const int gi = impl_->slot_group_index[(size_t)s];
    if (impl_->slot_is_win[(size_t)s])
        return impl_->win_scale_base_elems +
               (size_t)gi * ((size_t)impl_->win_total_blocks * impl_->elems_per_block /
                             impl_->cfg.head_dim);
    return (size_t)gi * impl_->scale_layer_stride;
}
bool KVCacheManager::windowed() const { return impl_->win_on; }
int KVCacheManager::window_pass_limit() const {
    return impl_->win_on ? impl_->win_pass_tokens : INT32_MAX;
}
bool KVCacheManager::prefix_sharing_supported() const { return !impl_->win_on; }
int* KVCacheManager::block_table_win(uint64_t seq_id) const {
    if (!impl_->win_on) return block_table(seq_id);
    auto it = impl_->seq_slot.find(seq_id);
    if (it == impl_->seq_slot.end()) return nullptr;
    return impl_->d_win_tables + (size_t)it->second * impl_->max_blocks_per_seq;
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
