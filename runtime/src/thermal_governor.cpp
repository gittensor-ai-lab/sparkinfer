// Thermally-adaptive decode pacing. See thermal_governor.h. The governor only ever *sleeps* —
// it cannot change what tokens are produced, only how fast. Temperature comes from the same
// engine-level query_gpu_stats() used for observability.

#include "sparkinfer/thermal_governor.h"
#include "sparkinfer/gpu_stats.h"

#include <chrono>
#include <thread>
#include <cstdio>

namespace sparkinfer {

using clk = std::chrono::steady_clock;
static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now().time_since_epoch()).count();
}

const char* ThermalGovernor::mode_name(Mode m) {
    switch (m) {
        case Mode::Turbo:     return "turbo";
        case Mode::Balanced:  return "balanced";
        case Mode::Safe:      return "safe";
        case Mode::Emergency: return "emergency";
    }
    return "?";
}

int ThermalGovernor::effective_temp_c(const Config& c, int temp_c, double slope_c_per_s) {
    if (temp_c < 0) return temp_c;
    int eff = temp_c;
    if (c.predict_horizon_ms > 0 && slope_c_per_s > 0.0)
        eff = temp_c + (int)(slope_c_per_s * c.predict_horizon_ms / 1000.0);
    return eff;
}

ThermalGovernor::Mode ThermalGovernor::classify(const Config& c, int temp_c) {
    if (temp_c >= c.emergency_c) return Mode::Emergency;
    if (temp_c >= c.safe_c)      return Mode::Safe;
    if (temp_c >= c.balanced_c)  return Mode::Balanced;
    return Mode::Turbo;
}

static double pace_ms_for(const ThermalGovernor::Config& c, ThermalGovernor::Mode m) {
    switch (m) {
        case ThermalGovernor::Mode::Balanced:  return c.balanced_ms;
        case ThermalGovernor::Mode::Safe:      return c.safe_ms;
        case ThermalGovernor::Mode::Emergency: return c.emergency_ms;
        default:                               return 0.0;  // Turbo
    }
}

double ThermalGovernor::pace() {
    if (!cfg_.enabled) return 0.0;

    const uint64_t t = now_ns();
    // Rate-limit sensor reads; reuse the cached temperature between samples.
    if (!started_ || (t - last_sample_ns_) >= (uint64_t)cfg_.sample_interval_ms * 1000000ull) {
        GpuStats g = query_gpu_stats(cfg_.device_id);
        if (g.valid && g.temp_c >= 0) {
            if (prev_sample_temp_ >= 0 && prev_sample_ns_ > 0) {
                const double dt_s = (double)(t - prev_sample_ns_) / 1e9;
                if (dt_s > 0.0) slope_ = (g.temp_c - prev_sample_temp_) / dt_s;
            }
            prev_sample_temp_ = g.temp_c;
            prev_sample_ns_   = t;
            last_temp_ = g.temp_c;
            if (g.temp_c > peak_temp_) peak_temp_ = g.temp_c;
        }
        last_sample_ns_ = t;
        started_ = true;
    }

    const Mode prev = mode_;
    int eff = last_temp_;
    if (cfg_.forced) {
        mode_ = cfg_.forced_mode;            // deterministic sweep: temperature ignored
    } else {
        if (last_temp_ < 0) return 0.0;      // auto mode + no sensor → never throttle
        eff = effective_temp_c(cfg_, last_temp_, slope_);
        mode_ = classify(cfg_, eff);
    }
    const double ms = pace_ms_for(cfg_, mode_);

    if (cfg_.log_transitions && mode_ != prev)
        fprintf(stderr, "[thermal] %d°C (slope %+.1f°C/s, eff %d) -> %s (%.0f ms/tok)\n",
                last_temp_, slope_, eff, mode_name(mode_), ms);

    if (ms > 0.0) {
        ++throttled_;
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
    }
    return ms;
}

} // namespace sparkinfer
