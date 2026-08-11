#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/lmcache_bridge_client.h"

namespace sparkinfer {

// Bytes needed for one chunk of `n_tok` tokens under `layout`, matching
// docs/lmcache_bridge_protocol.md's shm region layout (K bytes, V bytes, then K/V scale bytes
// only when layout.int8_kv). Both sides of the bridge derive this the same way -- see
// bridge/lmcache_bridge.py's ShmConnector.chunk_byte_size for the Python twin (which computes it
// purely to size an opaque byte buffer; it never interprets the internal (layer, block)
// sub-layout the way this side's stage_kv_to_shm/restore_kv_from_shm do).
size_t lmcache_chunk_byte_size(const BridgeKVLayout& layout, int n_tok);

// Copies seq_id's KV bytes for token range [start_tok, end_tok) into a freshly created shm
// region named shm_name (O_CREAT|O_EXCL -- fails if it already exists, callers pick unique
// names). end_tok - start_tok must be a positive multiple of layout.block_size; start_tok must
// also be block-aligned. Physical blocks are walked via kv.physical_block_ids() rather than
// assumed contiguous (sparkinfer's paging is free-list allocated). Blocking: synchronizes
// `stream` before returning so the region is fully populated by the time this returns.
//
// Returns false on any failure (bad range, shm create, cudaMemcpy) -- callers must treat that as
// "skip this store," never a crash; any partially-created shm region is cleaned up before
// returning false.
bool stage_kv_to_shm(KVCacheManager& kv, const BridgeKVLayout& layout, uint64_t seq_id,
                     int start_tok, int end_tok, const std::string& shm_name, cudaStream_t stream);

// Reverse of stage_kv_to_shm: copies bytes already sitting in shm_name at shm_offset_bytes (as
// reported by a LookupResult chunk) into seq_id's KV blocks for [start_tok, start_tok+len_tok).
// Caller must have already kv.allocate()'d enough blocks for seq_id to cover this range -- this
// function only fills bytes into blocks that already exist. Blocking: synchronizes `stream`
// before returning.
//
// Returns false on any failure (shm attach, out-of-range offset, cudaMemcpy) -- callers must
// treat a restore failure as a cache miss for this chunk (fall back to recompute), never a crash
// or corrupted-but-unreported KV state.
bool restore_kv_from_shm(KVCacheManager& kv, const BridgeKVLayout& layout, uint64_t seq_id,
                         const std::string& shm_name, uint64_t shm_offset_bytes, int start_tok,
                         int len_tok, cudaStream_t stream);

} // namespace sparkinfer
