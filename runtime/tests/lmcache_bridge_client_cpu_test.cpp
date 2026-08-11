// Exercises BridgeClient against a from-scratch fake peer implementing just enough of
// docs/lmcache_bridge_protocol.md's wire format to stand in for the real Python sidecar -- no
// GPU, no real lmcache, no shm KV staging (that's covered separately: shm_transfer.py/protocol.py
// by bridge/tests/, and the real round trip by bridge/tests/test_bridge_e2e.py). This test's job
// is BridgeClient's own socket/timeout/degradation behavior: does it handshake correctly, time
// out within budget instead of hanging, and survive a dead/absent/misbehaving peer without ever
// crashing or blocking indefinitely.
//
// The frame (de)serialization here is a deliberately independent, minimal reimplementation of
// the wire format (not a reuse of lmcache_bridge_client.cpp's private helpers) -- the point is to
// verify BridgeClient's actual on-the-wire behavior against a spec-compliant peer, not to share
// code with the thing being tested.
#include "sparkinfer/lmcache_bridge_client.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

using sparkinfer::BridgeClient;
using sparkinfer::BridgeKVLayout;
using sparkinfer::LookupResult;

constexpr uint32_t kMagic = 0x53494B56;
constexpr uint16_t kVersion = 1;
enum : uint16_t { HELLO = 1, HELLO_ACK = 2, LOOKUP = 3, LOOKUP_RESP = 4,
                  STORE = 5, STORE_ACK = 6, PING = 7, PONG = 8, ERROR_MSG = 9 };

#pragma pack(push, 1)
struct Header { uint32_t magic; uint16_t version; uint16_t msg_type; uint32_t payload_len; uint32_t request_id; };
#pragma pack(pop)
static_assert(sizeof(Header) == 16, "header size must match the protocol");

void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }
void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((uint8_t)(v >> (8 * i))); }
void put_str(std::vector<uint8_t>& b, const std::string& s) { put_u32(b, (uint32_t)s.size()); b.insert(b.end(), s.begin(), s.end()); }

std::vector<uint8_t> frame(uint16_t msg_type, uint32_t request_id, const std::vector<uint8_t>& payload) {
    Header h{kMagic, kVersion, msg_type, (uint32_t)payload.size(), request_id};
    std::vector<uint8_t> out((const uint8_t*)&h, (const uint8_t*)&h + sizeof(h));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// Minimal fake sidecar: accepts connections on `path`, and for each connection runs `handler`
// once per received frame (handler returns the raw response bytes to send back, or an empty
// vector to send nothing at all -- used to simulate a hung/non-responsive peer).
class FakePeer {
public:
    using Handler = std::function<std::vector<uint8_t>(uint16_t msg_type, uint32_t request_id,
                                                        const std::vector<uint8_t>& payload)>;

    FakePeer(std::string path, Handler handler) : path_(std::move(path)), handler_(std::move(handler)) {
        unlink(path_.c_str());
        fd_ = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        assert(fd_ >= 0);
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
        assert(bind(fd_, (struct sockaddr*)&addr, sizeof(addr)) == 0);
        assert(listen(fd_, 4) == 0);
        thread_ = std::thread([this] { run(); });
    }

    ~FakePeer() {
        running_ = false;
        shutdown(fd_, SHUT_RDWR);
        close(fd_);
        if (thread_.joinable()) thread_.join();
        unlink(path_.c_str());
    }

private:
    void run() {
        while (running_) {
            const int conn = accept(fd_, nullptr, nullptr);
            if (conn < 0) return;  // fd_ closed (destructor) or real error -- either way, stop
            std::thread([this, conn] {
                std::vector<uint8_t> buf(1 << 20);
                while (running_) {
                    const ssize_t n = recv(conn, buf.data(), buf.size(), 0);
                    if (n < (ssize_t)sizeof(Header)) break;
                    Header h;
                    memcpy(&h, buf.data(), sizeof(h));
                    std::vector<uint8_t> payload(buf.begin() + sizeof(Header), buf.begin() + n);
                    std::vector<uint8_t> resp = handler_(h.msg_type, h.request_id, payload);
                    if (!resp.empty()) send(conn, resp.data(), resp.size(), 0);
                }
                close(conn);
            }).detach();
        }
    }

    std::string path_;
    Handler handler_;
    int fd_ = -1;
    std::thread thread_;
    volatile bool running_ = true;
};

BridgeKVLayout test_layout() {
    BridgeKVLayout l;
    l.num_layers = 4;
    l.num_kv_heads = 2;
    l.head_dim = 8;
    l.block_size = 16;
    l.int8_kv = false;
    l.elem_bytes = 2;
    l.model_name = "test-model";
    return l;
}

std::string unique_socket_path(const char* tag) {
    return "/tmp/sparkinfer_lmcache_test_" + std::string(tag) + "_" + std::to_string(getpid()) + ".sock";
}

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main() {
    // lookup_timeout_ms_config() (lmcache_bridge_client.cpp) caches its env-var read in a
    // function-local static on first call -- a deliberate hot-path optimization for production
    // (read once, not per-lookup), but it means this test must set the value ONCE here, before
    // any BridgeClient::lookup() call anywhere below, rather than varying it per test case.
    // 500ms is generous enough that this from-scratch thread-per-connection fake peer (real
    // thread spawns + real socket round trips, no warm-path tuning) doesn't flake the happy-path
    // assertions below the way the production default (5ms, tuned for a warm same-host sidecar)
    // did during development of this test -- while still keeping the deliberately-hung-peer test
    // fast enough to run routinely.
    setenv("SPARKINFER_LMCACHE_LOOKUP_TIMEOUT_MS", "500", 1);

    // --- successful handshake + LOOKUP hit ---
    {
        const std::string path = unique_socket_path("hit");
        FakePeer peer(path, [](uint16_t msg_type, uint32_t rid, const std::vector<uint8_t>&) -> std::vector<uint8_t> {
            if (msg_type == HELLO) {
                std::vector<uint8_t> p;
                put_u8(p, 1);
                put_str(p, "0.0.0-fake");
                put_u32(p, 256);
                put_str(p, "");
                return frame(HELLO_ACK, rid, p);
            }
            if (msg_type == LOOKUP) {
                std::vector<uint8_t> p;
                put_u32(p, 256);       // matched_tokens (i32, but positive so byte layout is the same)
                put_str(p, "/sparkinfer_kv_test_hit");
                put_u32(p, 1);          // n_chunks
                put_u32(p, 0);          // start_tok
                put_u32(p, 256);        // len_tok
                for (int i = 0; i < 8; i++) p.push_back(0);  // shm_offset_bytes (u64) = 0
                return frame(LOOKUP_RESP, rid, p);
            }
            return {};
        });

        BridgeClient client(path, test_layout());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let the peer start accepting
        std::vector<int> tokens(256, 42);
        LookupResult res = client.lookup(tokens);
        assert(res.ok);
        assert(res.matched_tokens == 256);
        assert(res.shm_name == "/sparkinfer_kv_test_hit");
        assert(res.chunks.size() == 1);
        assert(res.chunks[0].start_tok == 0 && res.chunks[0].len_tok == 256);
        assert(client.is_alive());
        printf("[ok] handshake + LOOKUP hit\n");
    }

    // --- LOOKUP miss ---
    {
        const std::string path = unique_socket_path("miss");
        FakePeer peer(path, [](uint16_t msg_type, uint32_t rid, const std::vector<uint8_t>&) -> std::vector<uint8_t> {
            if (msg_type == HELLO) {
                std::vector<uint8_t> p;
                put_u8(p, 1); put_str(p, "0.0.0-fake"); put_u32(p, 256); put_str(p, "");
                return frame(HELLO_ACK, rid, p);
            }
            if (msg_type == LOOKUP) {
                std::vector<uint8_t> p;
                put_u32(p, 0); put_str(p, ""); put_u32(p, 0);
                return frame(LOOKUP_RESP, rid, p);
            }
            return {};
        });
        BridgeClient client(path, test_layout());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        LookupResult res = client.lookup(std::vector<int>(256, 1));
        assert(res.ok);
        assert(res.matched_tokens == 0);
        printf("[ok] LOOKUP miss\n");
    }

    // --- HELLO rejected (layout mismatch simulated by the fake peer) ---
    {
        const std::string path = unique_socket_path("reject");
        FakePeer peer(path, [](uint16_t msg_type, uint32_t rid, const std::vector<uint8_t>&) -> std::vector<uint8_t> {
            if (msg_type == HELLO) {
                std::vector<uint8_t> p;
                put_u8(p, 0); put_str(p, ""); put_u32(p, 0); put_str(p, "layout mismatch");
                return frame(HELLO_ACK, rid, p);
            }
            return {};
        });
        BridgeClient client(path, test_layout());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        LookupResult res = client.lookup(std::vector<int>(256, 1));
        assert(!res.ok);  // rejected handshake must degrade to a miss, not a crash or hang
        assert(!client.is_alive());
        printf("[ok] rejected HELLO degrades to miss\n");
    }

    // --- no peer listening at all: fast fail, must not block ---
    {
        const std::string path = unique_socket_path("nopeer");
        unlink(path.c_str());  // guarantee nothing is there
        BridgeClient client(path, test_layout());
        const auto t0 = std::chrono::steady_clock::now();
        LookupResult res = client.lookup(std::vector<int>(256, 1));
        const double ms = elapsed_ms(t0);
        assert(!res.ok);
        assert(ms < 200.0);  // connect() to a nonexistent AF_UNIX path fails immediately (ECONNREFUSED
                              // / ENOENT), not a timeout wait -- generous bound to absorb CI jitter
        printf("[ok] no peer listening -> fast fail (%.1fms)\n", ms);
    }

    // --- peer accepts but never responds to LOOKUP: must time out within budget, not hang ---
    {
        const std::string path = unique_socket_path("hang");
        FakePeer peer(path, [](uint16_t msg_type, uint32_t rid, const std::vector<uint8_t>&) -> std::vector<uint8_t> {
            if (msg_type == HELLO) {
                std::vector<uint8_t> p;
                put_u8(p, 1); put_str(p, "0.0.0-fake"); put_u32(p, 256); put_str(p, "");
                return frame(HELLO_ACK, rid, p);
            }
            return {};  // LOOKUP: silently swallowed, simulating a hung sidecar
        });
        // Reusing the 500ms budget set at the top of main() -- see the comment there for why
        // this can't be varied per test case (the timeout is read once and cached).
        BridgeClient client(path, test_layout());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto t0 = std::chrono::steady_clock::now();
        LookupResult res = client.lookup(std::vector<int>(256, 1));
        const double ms = elapsed_ms(t0);
        assert(!res.ok);
        assert(ms < 2000.0);  // bounded by the 500ms timeout (+ generous scheduling slack), never hangs
        printf("[ok] hung peer times out instead of hanging (%.1fms)\n", ms);
    }

    // --- malformed response: must degrade to a miss, not crash ---
    {
        const std::string path = unique_socket_path("malformed");
        FakePeer peer(path, [](uint16_t msg_type, uint32_t rid, const std::vector<uint8_t>&) -> std::vector<uint8_t> {
            if (msg_type == HELLO) {
                std::vector<uint8_t> p;
                put_u8(p, 1); put_str(p, "0.0.0-fake"); put_u32(p, 256); put_str(p, "");
                return frame(HELLO_ACK, rid, p);
            }
            if (msg_type == LOOKUP) {
                // Claims a LOOKUP_RESP with a huge payload_len but sends far fewer bytes --
                // BridgeClient's recv_frame_locked must reject this as truncated, not read OOB.
                std::vector<uint8_t> p;
                put_u32(p, 999999);  // matched_tokens: nonsensical, but that's not even reached --
                                     // the frame itself is short, caught before payload parsing
                Header h{kMagic, kVersion, LOOKUP_RESP, 999999, rid};  // payload_len lies
                std::vector<uint8_t> out((const uint8_t*)&h, (const uint8_t*)&h + sizeof(h));
                out.insert(out.end(), p.begin(), p.end());  // actual bytes sent: far short of 999999
                return out;
            }
            return {};
        });
        BridgeClient client(path, test_layout());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        LookupResult res = client.lookup(std::vector<int>(256, 1));
        assert(!res.ok);  // malformed frame -> miss, and critically: no crash getting here
        printf("[ok] malformed frame degrades to miss, no crash\n");
    }

    // --- STORE ack ---
    {
        const std::string path = unique_socket_path("store");
        std::atomic<bool> got_store{false};
        FakePeer peer(path, [&got_store](uint16_t msg_type, uint32_t rid, const std::vector<uint8_t>&) -> std::vector<uint8_t> {
            if (msg_type == HELLO) {
                std::vector<uint8_t> p;
                put_u8(p, 1); put_str(p, "0.0.0-fake"); put_u32(p, 256); put_str(p, "");
                return frame(HELLO_ACK, rid, p);
            }
            if (msg_type == STORE) {
                got_store = true;
                std::vector<uint8_t> p;
                put_u8(p, 1); put_str(p, "");
                return frame(STORE_ACK, rid, p);
            }
            return {};
        });
        // BridgeClient's store thread unlinks this shm name after the ack -- give it something
        // real to unlink so that codepath is exercised, not just skipped on ENOENT.
        const std::string shm_name = "/sparkinfer_kv_test_store_" + std::to_string(getpid());
        const int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
        assert(shm_fd >= 0);
        close(shm_fd);

        BridgeClient client(path, test_layout());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        client.store_async(std::vector<int>(256, 7), 0, 256, shm_name);
        // The STORE background thread does its own fresh handshake on a separate connection
        // (store_conn) with a 2000ms budget (hardcoded in lmcache_bridge_client.cpp -- STORE is
        // off the hot path, so it gets a more generous timeout than LOOKUP) before it can even
        // send the STORE frame -- poll comfortably past that worst case rather than the tighter
        // budget an earlier version of this test used, which flaked under scheduling jitter.
        for (int i = 0; i < 150 && !got_store; i++) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        assert(got_store.load());
        // Give the background thread a moment to process the ack and unlink.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        assert(shm_open(shm_name.c_str(), O_RDONLY, 0) < 0);  // unlinked by BridgeClient's store thread
        printf("[ok] STORE round trip + shm unlink\n");
    }

    printf("ALL PASSED\n");
    return 0;
}
