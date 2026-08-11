#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace sparkinfer {

// Layout the bridge needs to describe sparkinfer's KV to the sidecar in HELLO. Mirrors
// KVCacheConfig (kv_cache.h) plus the element size, which KVCacheConfig itself doesn't carry
// (int8_kv implies 1 byte, bf16 implies 2 -- BridgeClient wants it explicit rather than
// re-deriving the rule).
struct BridgeKVLayout {
    int num_layers = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    bool int8_kv = false;
    int elem_bytes = 2;
    std::string model_name;
};

// One matched chunk from a LOOKUP hit. shm_offset_bytes is this chunk's own offset into the
// shm region named by LookupResult::shm_name -- see docs/lmcache_bridge_protocol.md's "shm
// region layout" section. Each chunk is self-contained; chunks are not assumed contiguous.
struct BridgeKVChunk {
    int start_tok = 0;
    int len_tok = 0;
    uint64_t shm_offset_bytes = 0;
};

struct LookupResult {
    bool ok = false;   // false = miss, timeout, or bridge unavailable -- caller always falls
                       // back to normal recompute, this never distinguishes "why" to the caller
    int matched_tokens = 0;
    std::string shm_name;  // empty when matched_tokens == 0
    std::vector<BridgeKVChunk> chunks;
};

// C++ side of the sparkinfer <-> LMCache-sidecar bridge (docs/lmcache_bridge_protocol.md).
// Owns one AF_UNIX SOCK_SEQPACKET connection to the Python sidecar plus one dedicated background
// thread for STORE (so a slow STORE_ACK never stalls the caller). LOOKUP is synchronous with a
// short fixed timeout, called from ContinuousBatchEngine's single worker thread.
//
// Every method degrades safely: if the sidecar was never started, crashed, times out, or sends a
// malformed frame, LOOKUP returns a miss and STORE silently no-ops. Nothing here ever surfaces a
// bridge-down condition as a request-visible error -- see the "Degradation is always safe"
// invariant in the protocol doc.
class BridgeClient {
public:
    // socket_path: where the sidecar is listening (or will listen -- BridgeClient retries
    // connecting lazily, it does not require the sidecar to already be up at construction time).
    BridgeClient(std::string socket_path, BridgeKVLayout layout);
    ~BridgeClient();

    BridgeClient(const BridgeClient&) = delete;
    BridgeClient& operator=(const BridgeClient&) = delete;

    // True once HELLO/HELLO_ACK has succeeded and no failure has been observed since. Cheap
    // (atomic read) -- callers should check this before bothering to build a LOOKUP payload,
    // though lookup()/store_async() are themselves safe to call regardless.
    bool is_alive() const;

    // Synchronous, bounded by SPARKINFER_LMCACHE_LOOKUP_TIMEOUT_MS (default 5ms). Lazily
    // (re)connects/handshakes if the bridge isn't currently alive and the reconnect cooldown has
    // elapsed; a connect attempt that can't complete within the same budget also counts as a
    // miss for this call, it does not block the caller waiting for a slow connect.
    LookupResult lookup(const std::vector<int>& token_ids);

    // Fire-and-forget from the caller's perspective: hands the descriptor to the background
    // STORE thread and returns immediately. shm_name must already contain the staged KV bytes
    // (written by the caller before calling this) at the layout described in the protocol doc;
    // ownership of unlinking shm_name passes to the background thread once queued.
    void store_async(const std::vector<int>& token_ids, int new_start_tok, int new_end_tok,
                     std::string shm_name);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sparkinfer
