#pragma once

#include <cstdint>
#include <memory>

#include "sparkinfer/gpu_stats.h"

namespace sparkinfer {

struct RuntimeConfig {
    int device_id = 0;
    size_t kv_cache_bytes = 0;      // 0 = auto (80% of free VRAM)
    size_t expert_cache_bytes = 0;  // MoE expert residency budget
    int max_batch_size = 256;
    int max_seq_len = 32768;
    bool enable_cuda_graphs = true;
    bool enable_chunked_prefill = true;
};

class Runtime {
public:
    static std::unique_ptr<Runtime> create(const RuntimeConfig& cfg);

    virtual ~Runtime() = default;

    virtual void initialize() = 0;
    virtual void shutdown() = 0;

    // Returns device peak memory bandwidth in GB/s
    virtual float memory_bandwidth_gbps() const = 0;

    // Returns number of available CUDA SMs
    virtual int num_sms() const = 0;

    virtual int compute_capability_major() const = 0;
    virtual int compute_capability_minor() const = 0;

    // Engine-level GPU observability: a live sample of heat (°C) + VRAM (+ power/clock) on this
    // runtime's device. Safe to poll periodically (e.g. while decoding) to watch thermals/throttle.
    virtual GpuStats gpu_stats() const = 0;
};

} // namespace sparkinfer
