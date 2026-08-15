// Bulk KV <-> shm staging for the LMCache bridge. Physical KV blocks are free-list allocated
// (see kv_cache.cpp) and not guaranteed contiguous per sequence, so both directions walk
// (layer, physical_block) pairs individually rather than assuming one contiguous device range.
//
// Layout convention chosen here (layer-major, block-minor within each of K/V/K-scale/V-scale
// regions) is a purely internal agreement between stage_kv_to_shm and restore_kv_from_shm -- the
// Python sidecar never interprets these bytes (see bridge/lmcache_bridge.py's ShmConnector,
// which treats a chunk as an opaque blob of lmcache_chunk_byte_size() bytes), so nothing outside
// this file needs to agree with the ordering, only stage_kv_to_shm and restore_kv_from_shm need
// to agree with each other.
#include "sparkinfer/lmcache_staging.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cstring>

namespace sparkinfer {

namespace {

size_t elems_per_block(const BridgeKVLayout& layout) {
    return (size_t)layout.block_size * layout.num_kv_heads * layout.head_dim;
}
size_t scale_elems_per_block(const BridgeKVLayout& layout) {
    return (size_t)layout.block_size * layout.num_kv_heads;
}

}  // namespace

size_t lmcache_chunk_byte_size(const BridgeKVLayout& layout, int n_tok) {
    if (n_tok <= 0 || layout.block_size <= 0) return 0;
    const int n_blocks = n_tok / layout.block_size;
    const size_t block_bytes = elems_per_block(layout) * (size_t)layout.elem_bytes;
    size_t total = (size_t)layout.num_layers * 2 * (size_t)n_blocks * block_bytes;  // K + V
    if (layout.int8_kv) {
        const size_t scale_block_bytes = scale_elems_per_block(layout) * sizeof(uint16_t);
        total += (size_t)layout.num_layers * 2 * (size_t)n_blocks * scale_block_bytes;  // K+V scale
    }
    return total;
}

namespace {

// Copies one (K, V, or scale) region between a pinned host staging buffer and the device KV
// pool for n_blocks logical blocks starting at first_block, walking physical block ids so
// non-contiguous paging is handled correctly. `host_to_device` picks the copy direction; `off`
// (in/out) is the running byte offset within the staging buffer, advanced by this region's size
// regardless of success so the caller's next region starts at the right place even after a
// failure (the caller bails out on `ok=false` before that matters).
bool copy_region(void* pool, size_t layer_stride_elems, size_t region_elems_per_block,
                 int elem_bytes, const std::vector<int>& phys, int first_block, int n_blocks,
                 int num_layers, uint8_t* host_buf, size_t& off, bool host_to_device,
                 cudaStream_t stream) {
    const size_t nbytes = region_elems_per_block * (size_t)elem_bytes;
    for (int L = 0; L < num_layers; L++) {
        for (int b = 0; b < n_blocks; b++) {
            const int phys_block = phys[(size_t)(first_block + b)];
            const size_t dev_elem_off =
                (size_t)L * layer_stride_elems + (size_t)phys_block * region_elems_per_block;
            uint8_t* dev_ptr = (uint8_t*)pool + dev_elem_off * (size_t)elem_bytes;
            uint8_t* host_ptr = host_buf + off;
            off += nbytes;
            const cudaError_t err =
                host_to_device ? cudaMemcpyAsync(dev_ptr, host_ptr, nbytes, cudaMemcpyHostToDevice, stream)
                               : cudaMemcpyAsync(host_ptr, dev_ptr, nbytes, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) return false;
        }
    }
    return true;
}

}  // namespace

bool stage_kv_to_shm(KVCacheManager& kv, const BridgeKVLayout& layout, uint64_t seq_id,
                     int start_tok, int end_tok, const std::string& shm_name, cudaStream_t stream) {
    if (end_tok <= start_tok) return false;
    const int n_tok = end_tok - start_tok;
    if (layout.block_size <= 0 || start_tok % layout.block_size != 0 ||
        n_tok % layout.block_size != 0)
        return false;
    const int first_block = start_tok / layout.block_size;
    const int n_blocks = n_tok / layout.block_size;

    // A compacted (hybrid) pool holds one slice per FULL-ATTENTION layer, indexed by slot, so
    // the per-layer offsets below would walk past its end. Refuse rather than mis-index; this
    // path is not used by the hybrid models that compact.
    if (kv.kv_slots() != layout.num_layers) return false;
    const std::vector<int>& phys = kv.physical_block_ids(seq_id);
    if ((int)phys.size() < first_block + n_blocks) return false;

    const size_t total_bytes = lmcache_chunk_byte_size(layout, n_tok);
    if (total_bytes == 0) return false;

    void* h_staging = nullptr;
    if (cudaHostAlloc(&h_staging, total_bytes, cudaHostAllocDefault) != cudaSuccess) return false;
    uint8_t* host_buf = (uint8_t*)h_staging;
    size_t off = 0;
    bool ok = true;

    ok = ok && copy_region(kv.k_pool(), kv.layer_stride_elems(), elems_per_block(layout),
                           layout.elem_bytes, phys, first_block, n_blocks, layout.num_layers,
                           host_buf, off, /*host_to_device=*/false, stream);
    ok = ok && copy_region(kv.v_pool(), kv.layer_stride_elems(), elems_per_block(layout),
                           layout.elem_bytes, phys, first_block, n_blocks, layout.num_layers,
                           host_buf, off, false, stream);
    if (ok && layout.int8_kv) {
        ok = ok && copy_region(kv.k_scale_pool(), kv.scale_layer_stride_elems(),
                               scale_elems_per_block(layout), (int)sizeof(uint16_t), phys,
                               first_block, n_blocks, layout.num_layers, host_buf, off, false,
                               stream);
        ok = ok && copy_region(kv.v_scale_pool(), kv.scale_layer_stride_elems(),
                               scale_elems_per_block(layout), (int)sizeof(uint16_t), phys,
                               first_block, n_blocks, layout.num_layers, host_buf, off, false,
                               stream);
    }
    if (ok) ok = (cudaStreamSynchronize(stream) == cudaSuccess);
    if (!ok) {
        cudaFreeHost(h_staging);
        return false;
    }

#ifdef _WIN32
    // Named POSIX shm (shm_open/mmap) has no direct Windows equivalent wired up yet -- the LMCache
    // bridge is Linux-only for now (see lmcache_bridge_client.cpp's Windows stub); every caller of
    // stage_kv_to_shm already treats `false` as "bridge unavailable, fall back to recompute", so
    // this is a safe, fully-handled degradation rather than a crash or undefined behavior.
    cudaFreeHost(h_staging);
    return false;
#else
    const int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        cudaFreeHost(h_staging);
        return false;
    }
    if (ftruncate(fd, (off_t)total_bytes) != 0) {
        close(fd);
        shm_unlink(shm_name.c_str());
        cudaFreeHost(h_staging);
        return false;
    }
    void* mapped = mmap(nullptr, total_bytes, PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        shm_unlink(shm_name.c_str());
        cudaFreeHost(h_staging);
        return false;
    }
    memcpy(mapped, h_staging, total_bytes);
    munmap(mapped, total_bytes);
    cudaFreeHost(h_staging);
    return true;
#endif  // _WIN32
}

bool restore_kv_from_shm(KVCacheManager& kv, const BridgeKVLayout& layout, uint64_t seq_id,
                         const std::string& shm_name, uint64_t shm_offset_bytes, int start_tok,
                         int len_tok, cudaStream_t stream) {
    if (len_tok <= 0 || layout.block_size <= 0 || start_tok % layout.block_size != 0 ||
        len_tok % layout.block_size != 0)
        return false;
    const int first_block = start_tok / layout.block_size;
    const int n_blocks = len_tok / layout.block_size;

    // A compacted (hybrid) pool holds one slice per FULL-ATTENTION layer, indexed by slot, so
    // the per-layer offsets below would walk past its end. Refuse rather than mis-index; this
    // path is not used by the hybrid models that compact.
    if (kv.kv_slots() != layout.num_layers) return false;
    const std::vector<int>& phys = kv.physical_block_ids(seq_id);
    if ((int)phys.size() < first_block + n_blocks) return false;

    const size_t total_bytes = lmcache_chunk_byte_size(layout, len_tok);
    if (total_bytes == 0) return false;

#ifdef _WIN32
    return false;  // see the Windows note in stage_kv_to_shm above
#else
    const int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 ||
        shm_offset_bytes + total_bytes > (uint64_t)st.st_size) {
        close(fd);
        return false;
    }
    void* mapped = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return false;

    void* h_staging = nullptr;
    if (cudaHostAlloc(&h_staging, total_bytes, cudaHostAllocDefault) != cudaSuccess) {
        munmap(mapped, (size_t)st.st_size);
        return false;
    }
    memcpy(h_staging, (const uint8_t*)mapped + shm_offset_bytes, total_bytes);
    munmap(mapped, (size_t)st.st_size);

    uint8_t* host_buf = (uint8_t*)h_staging;
    size_t off = 0;
    bool ok = true;

    ok = ok && copy_region(kv.k_pool(), kv.layer_stride_elems(), elems_per_block(layout),
                           layout.elem_bytes, phys, first_block, n_blocks, layout.num_layers,
                           host_buf, off, /*host_to_device=*/true, stream);
    ok = ok && copy_region(kv.v_pool(), kv.layer_stride_elems(), elems_per_block(layout),
                           layout.elem_bytes, phys, first_block, n_blocks, layout.num_layers,
                           host_buf, off, true, stream);
    if (ok && layout.int8_kv) {
        ok = ok && copy_region(kv.k_scale_pool(), kv.scale_layer_stride_elems(),
                               scale_elems_per_block(layout), (int)sizeof(uint16_t), phys,
                               first_block, n_blocks, layout.num_layers, host_buf, off, true,
                               stream);
        ok = ok && copy_region(kv.v_scale_pool(), kv.scale_layer_stride_elems(),
                               scale_elems_per_block(layout), (int)sizeof(uint16_t), phys,
                               first_block, n_blocks, layout.num_layers, host_buf, off, true,
                               stream);
    }
    if (ok) ok = (cudaStreamSynchronize(stream) == cudaSuccess);

    cudaFreeHost(h_staging);
    return ok;
#endif  // _WIN32
}

} // namespace sparkinfer
