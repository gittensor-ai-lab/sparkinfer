// CPU test for the thermal governor's tiering policy — pure logic, no GPU needed.
// (classify() is a pure temp→mode map; a disabled governor never touches the device.)

#include "sparkinfer/thermal_governor.h"
#include <cstdio>
#include <string>

using G = sparkinfer::ThermalGovernor;
#define CHECK(x) do{ if(!(x)){ printf("FAIL: %s (line %d)\n", #x, __LINE__); return 1; } }while(0)

int main() {
    G::Config c;  // defaults: balanced 65, safe 70, emergency 80 °C

    // Reactive tiering at the boundaries.
    CHECK(G::classify(c, 50) == G::Mode::Turbo);
    CHECK(G::classify(c, 64) == G::Mode::Turbo);
    CHECK(G::classify(c, 65) == G::Mode::Balanced);
    CHECK(G::classify(c, 69) == G::Mode::Balanced);
    CHECK(G::classify(c, 70) == G::Mode::Safe);
    CHECK(G::classify(c, 79) == G::Mode::Safe);
    CHECK(G::classify(c, 80) == G::Mode::Emergency);
    CHECK(G::classify(c, 95) == G::Mode::Emergency);

    // Custom thresholds.
    G::Config c2; c2.balanced_c = 60; c2.safe_c = 72; c2.emergency_c = 85;
    CHECK(G::classify(c2, 59) == G::Mode::Turbo);
    CHECK(G::classify(c2, 60) == G::Mode::Balanced);
    CHECK(G::classify(c2, 71) == G::Mode::Balanced);
    CHECK(G::classify(c2, 72) == G::Mode::Safe);
    CHECK(G::classify(c2, 85) == G::Mode::Emergency);

    // A disabled governor is a strict no-op: never sleeps, stays in Turbo, touches no hardware.
    G off(c);
    CHECK(off.pace() == 0.0);
    CHECK(off.mode() == G::Mode::Turbo);
    CHECK(off.throttled_tokens() == 0);

    CHECK(std::string(G::mode_name(G::Mode::Turbo))     == "turbo");
    CHECK(std::string(G::mode_name(G::Mode::Emergency)) == "emergency");

    // Predictive projection bumps effective temp before crossing a threshold.
    G::Config cp;
    cp.balanced_c = 65; cp.predict_horizon_ms = 1500;
    CHECK(G::effective_temp_c(cp, 60, 0.0) == 60);
    CHECK(G::effective_temp_c(cp, 60, 4.0) == 66);   // 60 + 4*1.5
    CHECK(G::classify(cp, G::effective_temp_c(cp, 60, 4.0)) == G::Mode::Balanced);

    printf("thermal_governor_cpu_test: OK\n");
    return 0;
}
