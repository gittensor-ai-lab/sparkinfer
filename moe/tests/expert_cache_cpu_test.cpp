// CPU tests for expert cache sizing helpers (no GPU required).

#include "sparkinfer/moe/expert_cache.h"

#include <cstdio>
#include <cstdlib>

using sparkinfer::moe::expert_layer_bytes;
using sparkinfer::moe::max_resident_layers_from_bytes;
using sparkinfer::moe::normalize_moe_config;

static int failures = 0;

static void check(const char* name, bool ok) {
    if (!ok) { fprintf(stderr, "[FAIL] %s\n", name); failures++; }
    else     { printf("[PASS] %s\n", name); }
}

int main() {
    // Qwen3.5-ish: 256 experts, H=2048, F=512, bf16 gate+up+down
    const int E = 256, H = 2048, F = 512, layers = 40;
    const size_t layer_bytes = expert_layer_bytes(E, H, F);
    check("layer_bytes > 0", layer_bytes > 0);

    // ~1.5 GB per layer — 3 GB budget → 2 resident layers
    const size_t budget = layer_bytes * 2;
    const int resident = max_resident_layers_from_bytes(budget, E, H, F, layers);
    check("two layers fit in 2x budget", resident == 2);

    check("zero budget means all layers", max_resident_layers_from_bytes(0, E, H, F, layers) == layers);
    check("tiny budget clamps to 1", max_resident_layers_from_bytes(1024, E, H, F, layers) == 1);

    sparkinfer::moe::MoEConfig cfg{};
    cfg.num_experts = E;
    cfg.expert_cache_slots = 0;
    cfg = normalize_moe_config(cfg, 0);
    check("default slots = num_experts", cfg.expert_cache_slots == E);
    check("default prefetch depth", cfg.async_prefetch_depth == 2);

    return failures ? 1 : 0;
}
