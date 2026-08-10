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

    // Downgrade hysteresis: default config has hysteresis_c=3 (balanced=65, safe=70, emergency=80).
    {
        // Heating up is always immediate, regardless of the previous mode or hysteresis.
        CHECK(G::classify(c, 70, G::Mode::Turbo)    == G::Mode::Safe);
        CHECK(G::classify(c, 80, G::Mode::Balanced) == G::Mode::Emergency);

        // Cooling from Safe (entry 70): must drop below 70-3=67 to downgrade. A dip to 68 or 69
        // (still >= 67) stays in Safe instead of flapping back to Balanced.
        CHECK(G::classify(c, 69, G::Mode::Safe) == G::Mode::Safe);
        CHECK(G::classify(c, 68, G::Mode::Safe) == G::Mode::Safe);
        CHECK(G::classify(c, 67, G::Mode::Safe) == G::Mode::Safe);   // boundary: not yet below
        CHECK(G::classify(c, 66, G::Mode::Safe) == G::Mode::Balanced);

        // Cooling from Emergency (entry 80): stays Emergency until below 80-3=77, then jumps
        // straight to whatever tier the temperature naturally lands in.
        CHECK(G::classify(c, 78, G::Mode::Emergency) == G::Mode::Emergency);
        CHECK(G::classify(c, 76, G::Mode::Emergency) == G::Mode::Safe);
        CHECK(G::classify(c, 50, G::Mode::Emergency) == G::Mode::Turbo);

        // Unchanged temperature (natural == prev) is a no-op, not treated as heating or cooling.
        CHECK(G::classify(c, 72, G::Mode::Safe) == G::Mode::Safe);

        // hysteresis_c = 0 restores the legacy, state-free, symmetric behavior.
        G::Config c0 = c; c0.hysteresis_c = 0;
        CHECK(G::classify(c0, 69, G::Mode::Safe) == G::Mode::Balanced);
        CHECK(G::classify(c0, 66, G::Mode::Safe) == G::Mode::Balanced);
    }

    printf("thermal_governor_cpu_test: OK\n");
    return 0;
}
