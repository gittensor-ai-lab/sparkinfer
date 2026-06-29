#pragma once

#include <cstddef>
#include <string>

namespace sparkinfer {

// Engine-level GPU observability — a single on-demand sample of how hot the device is running
// and how much VRAM the model is resident in. Cheap enough to poll on a background thread while
// the engine decodes (see runtime/examples/qwen3_gguf_generate.cpp), so callers can track the
// PEAK heat/power a workload drives, not just a steady-state snapshot.
//
// VRAM always populates (cudaMemGetInfo). Temperature/power/clock come from NVML; if NVML isn't
// linked or the query fails, those stay -1 and `str()` simply omits them — never an error.
struct GpuStats {
    bool   valid            = false;  // false only if even VRAM couldn't be read
    int    temp_c           = -1;     // GPU core temperature, °C (heat)
    int    power_w          = -1;     // instantaneous board power draw, W
    int    sm_clock_mhz     = -1;     // SM clock, MHz (drops when thermal-throttling)
    size_t vram_used_bytes  = 0;      // resident VRAM = total - free
    size_t vram_total_bytes = 0;      // device VRAM capacity

    double vram_used_gb()  const { return vram_used_bytes  / 1e9; }
    double vram_total_gb() const { return vram_total_bytes / 1e9; }

    // e.g. "72°C · 21.4/31.4 GB VRAM · 410 W · 2820 MHz"
    std::string str() const;
};

// Sample heat + VRAM (+ power/clock) for a CUDA device. device_id < 0 ⇒ the current device.
// NVML is mapped to the CUDA device by PCI bus id, so this stays correct under CUDA_VISIBLE_DEVICES.
GpuStats query_gpu_stats(int device_id = -1);

} // namespace sparkinfer
