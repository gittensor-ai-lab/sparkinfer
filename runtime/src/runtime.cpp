// Runtime — device setup and capability query.

#include "sparkinfer/runtime.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace sparkinfer {

class RuntimeImpl : public Runtime {
public:
    explicit RuntimeImpl(const RuntimeConfig& cfg) : cfg_(cfg) {}

    void initialize() override {
        cudaError_t e = cudaSetDevice(cfg_.device_id);
        if (e != cudaSuccess) { fprintf(stderr, "[runtime] setDevice: %s\n", cudaGetErrorString(e)); return; }
        cudaDeviceProp p{};
        e = cudaGetDeviceProperties(&p, cfg_.device_id);
        if (e != cudaSuccess) { fprintf(stderr, "[runtime] getProps: %s\n", cudaGetErrorString(e)); return; }
        num_sms_ = p.multiProcessorCount;
        // bandwidth = 2 (DDR) * memClock(Hz) * busWidth(bytes) / 1e9.
        // memoryClockRate/memoryBusWidth were removed from cudaDeviceProp in CUDA 13;
        // query them via device attributes (portable across CUDA 12.x and 13.x).
        int mem_clock_khz = 0, bus_width_bits = 0;
        cudaDeviceGetAttribute(&mem_clock_khz,  cudaDevAttrMemoryClockRate,      cfg_.device_id);
        cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, cfg_.device_id);
        const double mem_hz = (double)mem_clock_khz * 1e3;
        const double bus_bytes = bus_width_bits / 8.0;
        bandwidth_gbps_ = (float)(2.0 * mem_hz * bus_bytes / 1e9);
        cc_major_ = p.major; cc_minor_ = p.minor;
        printf("[runtime] %s  cc=%d.%d  SMs=%d  BW=%.0f GB/s\n",
               p.name, p.major, p.minor, num_sms_, bandwidth_gbps_);
    }

    void shutdown() override { cudaDeviceSynchronize(); }
    float memory_bandwidth_gbps() const override { return bandwidth_gbps_; }
    int   num_sms() const override { return num_sms_; }
    GpuStats gpu_stats() const override { return query_gpu_stats(cfg_.device_id); }

private:
    RuntimeConfig cfg_;
    int num_sms_ = 0, cc_major_ = 0, cc_minor_ = 0;
    float bandwidth_gbps_ = 0.f;
};

std::unique_ptr<Runtime> Runtime::create(const RuntimeConfig& cfg) {
    return std::unique_ptr<Runtime>(new RuntimeImpl(cfg));
}

} // namespace sparkinfer
