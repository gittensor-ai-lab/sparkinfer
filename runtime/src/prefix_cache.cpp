#include "sparkinfer/prefix_cache.h"

#include <algorithm>
#include <utility>

namespace sparkinfer {

PrefixCache::PrefixCache(KVCacheManager* kv, const Limits& limits) : kv_(kv), limits_(limits) {}

PrefixCache::~PrefixCache() {
    std::lock_guard<std::mutex> lock(mu_);
    for (const Entry& e : entries_) kv_->release_blocks(e.blocks);
    entries_.clear();
}

PrefixCache::Hit PrefixCache::lookup(const std::vector<int>& prompt) {
    std::lock_guard<std::mutex> lock(mu_);
    Hit hit;
    stats_.lookups++;
    Entry* best = nullptr;
    for (Entry& e : entries_) {
        if (e.tokens.size() >= prompt.size()) continue;
        if (best && e.tokens.size() <= best->tokens.size()) continue;
        if (!std::equal(e.tokens.begin(), e.tokens.end(), prompt.begin())) continue;
        best = &e;
    }
    if (!best) return hit;
    best->last_used = ++clock_;
    stats_.hits++;
    stats_.tokens_reused += best->tokens.size();
    hit.tokens = (int)best->tokens.size();
    hit.blocks = best->blocks;
    hit.state = best->state;   // shares the pinned buffer; nothing ever writes to a stored snapshot
    return hit;
}

void PrefixCache::insert(std::vector<int> tokens, std::vector<int> blocks,
                         Qwen35Model::RecurrentStateSnapshot state) {
    std::lock_guard<std::mutex> lock(mu_);
    const size_t bs = (size_t)kv_->block_size();
    const bool well_formed = !tokens.empty() && bs > 0 && tokens.size() % bs == 0 &&
                             blocks.size() * bs == tokens.size();
    bool duplicate = false;
    for (Entry& e : entries_) {
        if (e.tokens.size() == tokens.size() && e.tokens == tokens) {
            e.last_used = ++clock_;
            duplicate = true;
            break;
        }
    }
    if (!well_formed || duplicate || state.bytes() > limits_.max_host_bytes) {
        kv_->release_blocks(blocks);
        return;
    }
    Entry e;
    e.tokens = std::move(tokens);
    e.blocks = std::move(blocks);
    e.state = std::move(state);
    e.last_used = ++clock_;
    host_bytes_ += e.state.bytes();
    for (int b : e.blocks) held_[b]++;
    entries_.push_back(std::move(e));
    stats_.inserts++;
    enforce_limits_locked();
}

bool PrefixCache::evict_for(int need_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    bool evicted = false;
    while (!entries_.empty() && kv_->num_free_blocks() < need_blocks) {
        evict_one_locked();
        evicted = true;
    }
    return evicted;
}

PrefixCache::Stats PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats s = stats_;
    s.entries = entries_.size();
    s.host_bytes = host_bytes_;
    s.blocks = (int)held_.size();
    return s;
}

void PrefixCache::evict_one_locked() {
    if (entries_.empty()) return;
    auto oldest = std::min_element(entries_.begin(), entries_.end(),
                                   [](const Entry& a, const Entry& b) { return a.last_used < b.last_used; });
    kv_->release_blocks(oldest->blocks);
    host_bytes_ -= oldest->state.bytes();
    for (int b : oldest->blocks) {
        auto it = held_.find(b);
        if (it != held_.end() && --it->second == 0) held_.erase(it);
    }
    entries_.erase(oldest);
    stats_.evictions++;
}

void PrefixCache::enforce_limits_locked() {
    while (!entries_.empty() &&
           (entries_.size() > limits_.max_entries || host_bytes_ > limits_.max_host_bytes ||
            (limits_.max_blocks > 0 && (int)held_.size() > limits_.max_blocks)))
        evict_one_locked();
}

}  // namespace sparkinfer
