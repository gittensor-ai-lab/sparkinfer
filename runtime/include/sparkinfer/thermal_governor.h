#pragma once

#include <cstdint>

namespace sparkinfer {

// Thermally-adaptive decode pacing — a software governor (DVFS-style, like a CPU frequency
// governor but for LLM decode). When the GPU runs hot it throttles decode THROUGHPUT by sleeping
// between tokens; it never touches weights, precision, logits, or sampling, so output is
// bit-identical to an un-paced run — just slower (and cooler) under thermal pressure.
//
// Levers it does NOT pull (would change the output): precision/quantization, model swap, KV
// truncation, speculative changes. The ONLY lever is an inter-token delay.
//
// Disabled by default: an enabled governor would slow a throughput benchmark, so the eval/bench
// path leaves it off (generate(..., gov=nullptr)) and pays zero overhead.
class ThermalGovernor {
public:
    enum class Mode { Turbo, Balanced, Safe, Emergency };

    struct Config {
        bool enabled    = false;   // master switch (default OFF)
        int  device_id  = -1;      // -1 = current CUDA device

        // Temperature thresholds (°C), ascending. A mode engages at >= its threshold.
        int  balanced_c  = 65;
        int  safe_c      = 70;
        int  emergency_c = 80;

        // Inter-token pace per mode (ms). Turbo is always 0 (no throttle). Larger = cooler+slower.
        double balanced_ms  = 2.0;
        double safe_ms      = 8.0;
        double emergency_ms = 25.0;

        // Re-read the sensor at most this often; the temperature is cached between reads so we
        // don't hit NVML every token (decode is ~ms/token, NVML is ~ms/call).
        int sample_interval_ms = 250;

        // Predictive throttle: if temperature is rising, project it `predict_horizon_ms` ahead and
        // tier on the projection, so we throttle BEFORE crossing a threshold. 0 = pure reactive.
        int predict_horizon_ms = 1500;

        bool log_transitions = false;   // print a line on each mode change (observability)
    };

    explicit ThermalGovernor(const Config& cfg) : cfg_(cfg) {}

    // Call once per decoded token. Rate-limited sensor read → mode update → sleep the mode's pace.
    // No-op (returns 0) when disabled or when no temperature is available. Returns ms slept.
    double pace();

    // Pure temperature→mode mapping (no hardware, no sleep) — the tiering policy, unit-testable.
    static Mode classify(const Config& cfg, int temp_c);
    static const char* mode_name(Mode m);

    Mode   mode()          const { return mode_; }
    int    last_temp_c()   const { return last_temp_; }
    int    peak_temp_c()   const { return peak_temp_; }
    double slope_c_per_s() const { return slope_; }
    uint64_t throttled_tokens() const { return throttled_; }  // tokens that got a non-zero pace

private:
    Config   cfg_;
    Mode     mode_       = Mode::Turbo;
    int      last_temp_  = -1;
    int      peak_temp_  = -1;
    double   slope_      = 0.0;       // °C/s, from consecutive samples
    uint64_t throttled_  = 0;
    // steady_clock bookkeeping (ns); set lazily on first pace()
    bool     started_         = false;
    uint64_t last_sample_ns_  = 0;
    int      prev_sample_temp_= -1;
    uint64_t prev_sample_ns_  = 0;
};

} // namespace sparkinfer
