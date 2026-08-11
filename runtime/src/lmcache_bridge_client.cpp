// C++ side of the sparkinfer <-> LMCache-sidecar bridge. Wire format is
// docs/lmcache_bridge_protocol.md -- keep both in sync by hand, there is no codegen here.
#include "sparkinfer/lmcache_bridge_client.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <mutex>

namespace sparkinfer {

namespace {

constexpr uint32_t kMagic = 0x53494B56;  // 'SIKV' little-endian on the wire
constexpr uint16_t kProtocolVersion = 1;
constexpr size_t kMaxFrameBytes = 2 * 1024 * 1024;  // see protocol doc's "Framing note"

enum MsgType : uint16_t {
    HELLO = 1,
    HELLO_ACK = 2,
    LOOKUP = 3,
    LOOKUP_RESP = 4,
    STORE = 5,
    STORE_ACK = 6,
    PING = 7,
    PONG = 8,
    ERROR_MSG = 9,
};

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t msg_type;
    uint32_t payload_len;
    uint32_t request_id;
};
#pragma pack(pop)
static_assert(sizeof(FrameHeader) == 16, "FrameHeader must match the 16-byte wire layout");

int lookup_timeout_ms_config() {
    static int ms = [] {
        const char* e = getenv("SPARKINFER_LMCACHE_LOOKUP_TIMEOUT_MS");
        int v = e ? atoi(e) : 5;
        return v > 0 ? v : 5;
    }();
    return ms;
}

// --- payload (de)serialization -------------------------------------------------------------

void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back((uint8_t)(v >> (8 * i)));
}
void put_i32(std::vector<uint8_t>& b, int32_t v) { put_u32(b, (uint32_t)v); }
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u32(b, (uint32_t)s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// Cursor-based reader over a received payload. Every getter bounds-checks and sets `ok=false`
// (sticky) on underflow instead of throwing -- a malformed frame degrades to "treat as failure",
// never a crash, matching the protocol doc's degradation invariant.
struct Reader {
    const uint8_t* p;
    size_t len;
    size_t off = 0;
    bool ok = true;

    bool need(size_t n) {
        if (!ok || off + n > len) { ok = false; return false; }
        return true;
    }
    uint8_t get_u8() {
        if (!need(1)) return 0;
        return p[off++];
    }
    uint32_t get_u32() {
        if (!need(4)) return 0;
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) v |= (uint32_t)p[off + i] << (8 * i);
        off += 4;
        return v;
    }
    uint64_t get_u64() {
        if (!need(8)) return 0;
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v |= (uint64_t)p[off + i] << (8 * i);
        off += 8;
        return v;
    }
    int32_t get_i32() { return (int32_t)get_u32(); }
    std::string get_str() {
        uint32_t n = get_u32();
        if (!ok || !need(n)) return {};
        std::string s((const char*)p + off, n);
        off += n;
        return s;
    }
};

// One AF_UNIX SOCK_SEQPACKET connection with its own liveness/reconnect state. ctrl (HELLO +
// LOOKUP + PING) and store_conn (HELLO + STORE) are deliberately separate instances -- see the
// "Transport" section of the protocol doc for why a slow STORE must never delay a LOOKUP.
// Namespace-scoped (not nested in BridgeClient) purely so it can be defined here without also
// forward-declaring it in the public header alongside Impl.
struct Connection {
    std::mutex mu;
    int fd = -1;
    std::atomic<bool> alive{false};
    std::chrono::steady_clock::time_point next_attempt{};  // cooldown gate, epoch = never waited
    int backoff_ms = 5000;
    static constexpr int kMaxBackoffMs = 60000;
    // Reused across every recv on this connection so LOOKUP's tight timeout budget never pays
    // for a fresh 2MiB allocation + zero-fill on every call -- sized lazily on first use, then
    // kept for the connection's lifetime (reconnects don't need to resize it).
    std::vector<uint8_t> recv_buf;

    void close_locked() {
        if (fd >= 0) { ::close(fd); fd = -1; }
        alive.store(false, std::memory_order_relaxed);
    }
};

}  // namespace

struct BridgeClient::Impl {
    std::string socket_path;
    BridgeKVLayout layout;

    Connection ctrl;
    Connection store_conn;

    std::atomic<uint32_t> next_request_id{1};

    std::atomic<bool> running{true};
    std::thread ping_thread;
    std::thread store_thread;

    struct StoreItem {
        std::vector<int> token_ids;
        int new_start = 0;
        int new_end = 0;
        std::string shm_name;
    };
    std::mutex store_mu;
    std::condition_variable store_cv;
    std::deque<StoreItem> store_queue;

    // Dedicated to the ping thread's interruptible sleep -- deliberately NOT store_cv/store_mu.
    // An earlier version shared them (as a ready-made shutdown signal, since both threads just
    // needed "wake up when running goes false"), but std::condition_variable::notify_one() wakes
    // an arbitrary one of ALL current waiters, not a specific one: store_async()'s notify_one()
    // could wake the ping thread instead of the store thread, leaving a queued STORE stuck until
    // something else happened to wake the right thread (found via this file's own unit test
    // flaking -- a queued store sometimes never got sent within a many-second window, not just a
    // slow one). Two independent condition variables makes that class of bug structurally
    // impossible rather than relying on both predicates happening to be safe under notify_all.
    std::mutex shutdown_mu;
    std::condition_variable shutdown_cv;
};

namespace {

bool set_timeout(int fd, int ms) {
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    bool ok = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
    ok = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0 && ok;
    return ok;
}

// Connects a fresh fd for `conn` (closing any prior one), honoring the cooldown -- returns false
// without touching the socket at all if still in cooldown. Does not send HELLO; caller does that
// so it can use its own timeout budget for the handshake round trip.
bool dial_locked(Connection& conn, const std::string& path, int timeout_ms) {
    const auto now = std::chrono::steady_clock::now();
    if (now < conn.next_attempt) return false;

    conn.close_locked();

    int fd = ::socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (fd < 0) {
        conn.next_attempt = now + std::chrono::milliseconds(conn.backoff_ms);
        conn.backoff_ms = std::min(conn.backoff_ms * 2, Connection::kMaxBackoffMs);
        return false;
    }
    set_timeout(fd, timeout_ms);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        return false;
    }
    memcpy(addr.sun_path, path.c_str(), path.size() + 1);

    if (::connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        ::close(fd);
        conn.next_attempt = now + std::chrono::milliseconds(conn.backoff_ms);
        conn.backoff_ms = std::min(conn.backoff_ms * 2, Connection::kMaxBackoffMs);
        return false;
    }

    conn.fd = fd;
    return true;
}

// One send() call carrying header+payload as a single SOCK_SEQPACKET message -- see the
// protocol doc's "Framing note": splitting this into two send()s would arrive as two messages.
bool send_frame_locked(Connection& conn, uint16_t msg_type, uint32_t request_id,
                       const std::vector<uint8_t>& payload) {
    if (conn.fd < 0) return false;
    std::vector<uint8_t> buf;
    buf.reserve(sizeof(FrameHeader) + payload.size());
    FrameHeader h{kMagic, kProtocolVersion, msg_type, (uint32_t)payload.size(), request_id};
    buf.insert(buf.end(), (const uint8_t*)&h, (const uint8_t*)&h + sizeof(h));
    buf.insert(buf.end(), payload.begin(), payload.end());
    const ssize_t n = ::send(conn.fd, buf.data(), buf.size(), 0);
    return n == (ssize_t)buf.size();
}

// One recv() call retrieving exactly one message. Returns false on timeout, short read, bad
// magic, or a truncated (oversized) message -- any of these is treated as a failed request by
// the caller, never a crash.
bool recv_frame_locked(Connection& conn, FrameHeader* out_header,
                       std::vector<uint8_t>* out_payload) {
    if (conn.fd < 0) return false;
    if (conn.recv_buf.empty()) conn.recv_buf.resize(kMaxFrameBytes);
    const ssize_t n = ::recv(conn.fd, conn.recv_buf.data(), conn.recv_buf.size(), 0);
    if (n < (ssize_t)sizeof(FrameHeader)) return false;
    FrameHeader h;
    memcpy(&h, conn.recv_buf.data(), sizeof(h));
    if (h.magic != kMagic) return false;
    if (sizeof(FrameHeader) + h.payload_len != (size_t)n) return false;  // short/truncated msg
    *out_header = h;
    out_payload->assign(conn.recv_buf.begin() + sizeof(FrameHeader), conn.recv_buf.begin() + n);
    return true;
}

std::vector<uint8_t> build_hello_payload(const BridgeKVLayout& layout) {
    std::vector<uint8_t> p;
    put_u32(p, (uint32_t)layout.num_layers);
    put_u32(p, (uint32_t)layout.num_kv_heads);
    put_u32(p, (uint32_t)layout.head_dim);
    put_u32(p, (uint32_t)layout.block_size);
    put_u8(p, layout.int8_kv ? 1 : 0);
    put_u32(p, (uint32_t)layout.elem_bytes);
    put_str(p, layout.model_name);
    return p;
}

// Connect (respecting cooldown) + HELLO/HELLO_ACK, all within `timeout_ms`. On any failure the
// connection is left closed and the cooldown/backoff from dial_locked applies; a version
// mismatch or ok=0 ack is treated identically to a transport failure -- callers never retry the
// handshake on the same fd, per the protocol doc.
bool handshake_locked(Connection& conn, const std::string& path,
                      const BridgeKVLayout& layout, uint32_t request_id, int timeout_ms) {
    if (conn.alive.load(std::memory_order_relaxed)) return true;
    if (!dial_locked(conn, path, timeout_ms)) return false;

    if (!send_frame_locked(conn, HELLO, request_id, build_hello_payload(layout))) {
        conn.close_locked();
        return false;
    }
    FrameHeader h{};
    std::vector<uint8_t> payload;
    if (!recv_frame_locked(conn, &h, &payload) || h.msg_type != HELLO_ACK ||
        h.version != kProtocolVersion) {
        conn.close_locked();
        return false;
    }
    Reader r{payload.data(), payload.size()};
    const uint8_t ok = r.get_u8();
    (void)r.get_str();  // lmcache_version, informational only
    (void)r.get_u32();  // chunk_size_tokens, informational only for the C++ side (the sidecar is
                        // the one that enforces chunk alignment)
    (void)r.get_str();  // error, only meaningful when ok==0
    if (!r.ok || !ok) {
        conn.close_locked();
        return false;
    }
    conn.alive.store(true, std::memory_order_relaxed);
    conn.backoff_ms = 5000;  // reset backoff on a successful handshake
    return true;
}

}  // namespace

BridgeClient::BridgeClient(std::string socket_path, BridgeKVLayout layout) : impl_(new Impl()) {
    impl_->socket_path = std::move(socket_path);
    impl_->layout = std::move(layout);

    impl_->ping_thread = std::thread([this] {
        while (impl_->running.load(std::memory_order_relaxed)) {
            // Interruptible sleep on its own dedicated condition variable (not store_cv -- see
            // Impl::shutdown_cv's comment for why sharing one with the STORE thread is a real
            // bug, not just a style choice) so ~BridgeClient() doesn't block for up to 2s waiting
            // on a plain sleep_for() to notice running=false.
            {
                std::unique_lock<std::mutex> wait_lock(impl_->shutdown_mu);
                impl_->shutdown_cv.wait_for(wait_lock, std::chrono::milliseconds(2000),
                                            [this] { return !impl_->running.load(std::memory_order_relaxed); });
            }
            if (!impl_->running.load(std::memory_order_relaxed)) break;
            std::lock_guard<std::mutex> lock(impl_->ctrl.mu);
            const uint32_t rid = impl_->next_request_id.fetch_add(1);
            // Short budget: a hung PING must not tie up ctrl for long, since LOOKUP shares it.
            if (!handshake_locked(impl_->ctrl, impl_->socket_path, impl_->layout, rid, 50))
                continue;
            if (!send_frame_locked(impl_->ctrl, PING, rid, {})) {
                impl_->ctrl.close_locked();
                continue;
            }
            FrameHeader h{};
            std::vector<uint8_t> payload;
            if (!recv_frame_locked(impl_->ctrl, &h, &payload) || h.msg_type != PONG)
                impl_->ctrl.close_locked();
        }
    });

    impl_->store_thread = std::thread([this] {
        while (true) {
            Impl::StoreItem item;
            {
                std::unique_lock<std::mutex> lock(impl_->store_mu);
                impl_->store_cv.wait(lock, [this] {
                    return !impl_->store_queue.empty() ||
                           !impl_->running.load(std::memory_order_relaxed);
                });
                if (impl_->store_queue.empty()) {
                    if (!impl_->running.load(std::memory_order_relaxed)) return;
                    continue;
                }
                item = std::move(impl_->store_queue.front());
                impl_->store_queue.pop_front();
            }

            std::vector<uint8_t> payload;
            put_u32(payload, (uint32_t)item.token_ids.size());
            for (int t : item.token_ids) put_i32(payload, t);
            put_u32(payload, (uint32_t)item.new_start);
            put_u32(payload, (uint32_t)item.new_end);
            put_str(payload, item.shm_name);

            bool acked = false;
            {
                std::lock_guard<std::mutex> lock(impl_->store_conn.mu);
                const uint32_t rid = impl_->next_request_id.fetch_add(1);
                // STORE is off the hot path; a generous budget is fine here, unlike LOOKUP/PING.
                if (handshake_locked(impl_->store_conn, impl_->socket_path, impl_->layout, rid,
                                     2000) &&
                    send_frame_locked(impl_->store_conn, STORE, rid, payload)) {
                    FrameHeader h{};
                    std::vector<uint8_t> resp;
                    if (recv_frame_locked(impl_->store_conn, &h, &resp) && h.msg_type == STORE_ACK) {
                        Reader r{resp.data(), resp.size()};
                        acked = r.get_u8() != 0;
                    } else {
                        impl_->store_conn.close_locked();
                    }
                }
            }
            // The C++ side created shm_name for this STORE (per the protocol doc's shm
            // lifecycle) -- unlink it now regardless of ack outcome. On a failed/timed-out
            // STORE the sidecar never mapped it, so this is just cleaning up our own staging
            // buffer; on success the sidecar has already copied out of it by the time STORE_ACK
            // arrives (synchronous within its own handler).
            if (!item.shm_name.empty()) shm_unlink(item.shm_name.c_str());
            (void)acked;  // no caller to report to -- store_async() is fire-and-forget by design
        }
    });
}

BridgeClient::~BridgeClient() {
    impl_->running.store(false, std::memory_order_relaxed);
    impl_->store_cv.notify_all();
    impl_->shutdown_cv.notify_all();
    if (impl_->ping_thread.joinable()) impl_->ping_thread.join();
    if (impl_->store_thread.joinable()) impl_->store_thread.join();
    {
        std::lock_guard<std::mutex> lock(impl_->ctrl.mu);
        impl_->ctrl.close_locked();
    }
    {
        std::lock_guard<std::mutex> lock(impl_->store_conn.mu);
        impl_->store_conn.close_locked();
    }
}

bool BridgeClient::is_alive() const {
    return impl_->ctrl.alive.load(std::memory_order_relaxed);
}

LookupResult BridgeClient::lookup(const std::vector<int>& token_ids) {
    LookupResult out;
    const int timeout_ms = lookup_timeout_ms_config();

    std::lock_guard<std::mutex> lock(impl_->ctrl.mu);
    const uint32_t rid = impl_->next_request_id.fetch_add(1);
    if (!handshake_locked(impl_->ctrl, impl_->socket_path, impl_->layout, rid, timeout_ms))
        return out;  // ok=false: bridge unavailable, caller recomputes -- never blocks longer
                     // than one handshake attempt at `timeout_ms`

    std::vector<uint8_t> payload;
    put_u32(payload, (uint32_t)token_ids.size());
    for (int t : token_ids) put_i32(payload, t);

    if (!send_frame_locked(impl_->ctrl, LOOKUP, rid, payload)) {
        impl_->ctrl.close_locked();
        return out;
    }
    FrameHeader h{};
    std::vector<uint8_t> resp;
    if (!recv_frame_locked(impl_->ctrl, &h, &resp)) {
        // Timeout or malformed response: leave the connection open (this is not necessarily a
        // dead bridge, just a slow one this time) but return a miss for this call -- the
        // worker thread's budget is what matters, not whether the sidecar eventually replies.
        return out;
    }
    if (h.msg_type == ERROR_MSG) return out;  // sidecar-reported failure: miss, not a crash
    if (h.msg_type != LOOKUP_RESP) {
        impl_->ctrl.close_locked();  // genuinely unexpected framing: treat as connection-broken
        return out;
    }

    Reader r{resp.data(), resp.size()};
    out.matched_tokens = (int)r.get_i32();
    out.shm_name = r.get_str();
    // Not reserve()'d against n_chunks: it comes straight off the wire, unvalidated, and a
    // malformed/corrupt frame could claim an enormous count -- push_back's incremental growth
    // is bounded by how many chunks actually fit in the payload (r.ok flips false the moment a
    // read runs past the buffer), reserve() would not be.
    const uint32_t n_chunks = r.get_u32();
    for (uint32_t i = 0; i < n_chunks && r.ok; i++) {
        BridgeKVChunk c;
        c.start_tok = (int)r.get_i32();
        c.len_tok = (int)r.get_i32();
        c.shm_offset_bytes = r.get_u64();
        out.chunks.push_back(c);
    }
    if (!r.ok) return LookupResult{};  // malformed payload: treat the whole response as a miss

    out.ok = true;
    return out;
}

void BridgeClient::store_async(const std::vector<int>& token_ids, int new_start_tok,
                               int new_end_tok, std::string shm_name) {
    std::lock_guard<std::mutex> lock(impl_->store_mu);
    impl_->store_queue.push_back(
        Impl::StoreItem{token_ids, new_start_tok, new_end_tok, std::move(shm_name)});
    impl_->store_cv.notify_one();
}

} // namespace sparkinfer
