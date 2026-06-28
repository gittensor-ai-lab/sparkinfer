// Engine-level GPU observability: heat (°C) + VRAM (+ power/clock).
//
// VRAM is read with cudaMemGetInfo (always available). Temperature/power/clock use NVML when it's
// linked (SPARKINFER_HAVE_NVML, set by CMake when CUDA::nvml exists). NVML is initialized once and
// left open for the process so repeated background sampling stays cheap.

#include "sparkinfer/gpu_stats.h"

#include <cuda_runtime.h>
#include <cstdio>

#ifdef SPARKINFER_HAVE_NVML
#include <nvml.h>
#endif

namespace sparkinfer {

#ifdef SPARKINFER_HAVE_NVML
// Init NVML once (not per sample); never shut down — process exit reclaims it.
static bool nvml_ready() {
    static bool ok = (nvmlInit_v2() == NVML_SUCCESS);
    return ok;
}
#endif

std::string GpuStats::str() const {
    if (!valid) return "GPU stats unavailable";
    char buf[192];
    const int cap = (int)sizeof buf;
    // snprintf returns the length it WOULD have written; on truncation that exceeds the
    // buffer, so advancing `buf + n` with `cap - n` (size_t) would point past the buffer
    // and underflow the size. Append() clamps the cursor to what was actually written.
    int n = 0;
    auto append = [&](const char* fmt, auto... args) {
        if (n >= cap - 1) return;
        int m = snprintf(buf + n, (size_t)(cap - n), fmt, args...);
        if (m < 0) return;
        n += (m < cap - 1 - n) ? m : cap - 1 - n;   // clamp to bytes actually written
    };
    append("%d°C · %.1f/%.1f GB VRAM", temp_c, vram_used_gb(), vram_total_gb());
    if (power_w      >= 0) append(" · %d W", power_w);
    if (sm_clock_mhz >= 0) append(" · %d MHz", sm_clock_mhz);
    return std::string(buf, n);
}

GpuStats query_gpu_stats(int device_id) {
    GpuStats s;

    int dev = device_id;
    if (dev < 0) cudaGetDevice(&dev);

    // VRAM — cudaMemGetInfo reports the *current* device, so switch to `dev` and restore.
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != dev) cudaSetDevice(dev);
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) == cudaSuccess && totb > 0) {
        s.vram_total_bytes = totb;
        s.vram_used_bytes  = (totb >= freeb) ? (totb - freeb) : 0;
        s.valid = true;
    }
    if (prev != dev && prev >= 0) cudaSetDevice(prev);

#ifdef SPARKINFER_HAVE_NVML
    // Heat/power/clock via NVML. Resolve the handle by PCI bus id so the CUDA device and the NVML
    // device agree even when CUDA_VISIBLE_DEVICES remaps ordinals.
    char pci[32] = {0};
    if (nvml_ready() && cudaDeviceGetPCIBusId(pci, sizeof pci, dev) == cudaSuccess) {
        nvmlDevice_t h{};
        if (nvmlDeviceGetHandleByPciBusId_v2(pci, &h) == NVML_SUCCESS) {
            unsigned t = 0, p = 0, c = 0;
            // These NVML getters are marked deprecated in CUDA 13's nvml.h but remain the portable
            // way to read temp/power across CUDA 12 and 13 — silence the deprecation note.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            if (nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS) s.temp_c = (int)t;
            if (nvmlDeviceGetPowerUsage(h, &p) == NVML_SUCCESS)                        s.power_w = (int)(p / 1000); // mW→W
            if (nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM, &c) == NVML_SUCCESS)          s.sm_clock_mhz = (int)c;
#pragma GCC diagnostic pop
            s.valid = true;
        }
    }
#endif
    return s;
}

} // namespace sparkinfer
