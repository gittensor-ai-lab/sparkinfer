#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>

namespace sparkinfer_server {

class RequestSlots {
public:
    explicit RequestSlots(int max_slots) : max_slots_(max_slots), in_use_(0) {}

    // Default of 2 matches the KV cache pool sizing in model_engine.cpp — both
    // read the same env var so they can never drift out of sync with each
    // other. See the comment there for why the pool has to scale with this.
    static int from_env(int default_slots = 2) {
        const char* e = std::getenv("SPARKINFER_MAX_CONCURRENT");
        if (!e || !*e) return default_slots;
        const int v = std::atoi(e);
        return v > 0 ? v : default_slots;
    }

    bool try_acquire() {
        std::lock_guard<std::mutex> lock(mu_);
        if (in_use_ >= max_slots_) return false;
        ++in_use_;
        return true;
    }

    // Blocks until a slot frees up or timeout_ms elapses. Returns false on
    // timeout — caller should respond with a "busy" error rather than
    // proceed, not silently run over capacity.
    bool acquire(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mu_);
        const bool got_slot =
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return in_use_ < max_slots_; });
        if (!got_slot) return false;
        ++in_use_;
        return true;
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (in_use_ > 0) --in_use_;
        }
        cv_.notify_one();
    }

    int max_slots() const { return max_slots_; }
    int in_use() const {
        std::lock_guard<std::mutex> lock(mu_);
        return in_use_;
    }

private:
    const int max_slots_;
    int in_use_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
};

// Move-only RAII release guard for a slot acquired via RequestSlots::acquire()/
// try_acquire(). Copy is explicitly deleted — a naive default move (member-wise
// pointer copy, source left pointing at the same pool) would double-release
// when both the moved-from and moved-to instances are destroyed. Held via
// shared_ptr at call sites that need to capture it into a copyable
// std::function (httplib's chunked content provider), since std::function
// requires its target to be copy-constructible even if only ever invoked
// once — the shared_ptr's single underlying SlotLease still only releases
// once, when the last copy goes away.
struct SlotLease {
    RequestSlots* pool = nullptr;

    SlotLease() = default;
    explicit SlotLease(RequestSlots* p) : pool(p) {}
    SlotLease(const SlotLease&) = delete;
    SlotLease& operator=(const SlotLease&) = delete;

    SlotLease(SlotLease&& other) noexcept : pool(other.pool) { other.pool = nullptr; }
    SlotLease& operator=(SlotLease&& other) noexcept {
        if (this != &other) {
            if (pool) pool->release();
            pool = other.pool;
            other.pool = nullptr;
        }
        return *this;
    }

    ~SlotLease() {
        if (pool) pool->release();
    }
};

}  // namespace sparkinfer_server
