#pragma once

// Automatic prefix cache for ContinuousBatchEngine.
//
// Chat and agent clients resend the whole conversation on every turn, so each request's prompt is
// the previous request's prompt plus what happened since. Without a cache the server recomputes
// all of it. An entry here is a prefix that was already computed: its KV blocks, shared with any
// later sequence that starts from it (KVCacheManager's refcounted blocks, no copy), and a snapshot
// of the Gated-DeltaNet recurrent state at the end of it.
//
// The snapshot is why an entry is taken at ONE chosen position rather than at every block. KV can
// be cut at any block boundary after the fact, but a recurrent layer's state is a single running
// value -- it exists only at the position the sequence is at, and copying it costs ~205 MB of
// pinned host memory on Qwen3.8-27B. So the engine snapshots each request at a checkpoint the
// caller picks (the server picks the start of the final assistant turn, which is where the next
// turn's prompt stops matching this one), and the cache keeps the most recently used of those.
//
// Every method that touches blocks must run under the model's device mutex, the same lock every
// other KVCacheManager mutation already takes. The cache's own mutex only guards its bookkeeping,
// so stats() can be read from an HTTP thread.

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

class PrefixCache {
public:
    struct Limits {
        size_t max_entries = 32;
        size_t max_host_bytes = size_t(8) << 30;   // pinned host memory for recurrent snapshots
        int max_blocks = 0;                        // distinct KV blocks entries may hold; 0 = no cap
    };

    struct Hit {
        int tokens = 0;                            // cached prefix length; 0 on a miss
        std::vector<int> blocks;                   // tokens / block_size physical blocks to share
        Qwen35Model::RecurrentStateSnapshot state;
    };

    struct Stats {
        uint64_t lookups = 0;
        uint64_t hits = 0;
        uint64_t tokens_reused = 0;
        uint64_t inserts = 0;
        uint64_t evictions = 0;
        size_t entries = 0;
        size_t host_bytes = 0;
        int blocks = 0;          // distinct KV blocks held: entries on one conversation share most
    };

    PrefixCache(KVCacheManager* kv, const Limits& limits);
    ~PrefixCache();   // releases every block the entries hold
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    // The longest entry that is a PROPER prefix of `prompt` -- strictly shorter, so at least one
    // prompt token is still prefilled and produces the first generated token. Refreshes its LRU
    // position. The returned blocks are not retained by the hit; share them before anything can
    // evict (both happen under the device mutex).
    Hit lookup(const std::vector<int>& prompt);

    // Adopt `blocks` (already retained for this entry) and `state` as the entry for `tokens`, whose
    // length must be a whole number of blocks. A prefix already present, or a malformed entry, is
    // dropped and its blocks released. Evicts least-recently-used entries past the limits.
    void insert(std::vector<int> tokens, std::vector<int> blocks,
                Qwen35Model::RecurrentStateSnapshot state);

    // Evict least-recently-used entries until the pool has at least need_blocks free or the cache
    // is empty. True if anything was evicted. Blocks still shared with a live sequence do not
    // return to the pool until that sequence finishes, so this can empty the cache and still fall
    // short.
    bool evict_for(int need_blocks);

    Stats stats() const;

private:
    struct Entry {
        std::vector<int> tokens;
        std::vector<int> blocks;
        Qwen35Model::RecurrentStateSnapshot state;
        uint64_t last_used = 0;
    };

    void evict_one_locked();
    void enforce_limits_locked();

    KVCacheManager* kv_;
    Limits limits_;
    mutable std::mutex mu_;
    std::vector<Entry> entries_;
    uint64_t clock_ = 0;
    Stats stats_;
    size_t host_bytes_ = 0;
    // Entries holding each block. Entries along one conversation share most of their blocks, so a
    // cap on the sum would evict for memory that is not actually used twice.
    std::unordered_map<int, int> held_;
};

}  // namespace sparkinfer
