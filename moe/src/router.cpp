// Router — thin host wrapper over the device top-k router kernel.
// In sync-free mode the per-expert counts stay on the GPU and are not copied
// back; last_expert_counts() returns empty so callers can't accidentally insert
// a host sync that would break CUDA-graph capture.

#include "sparkinfer/moe/router.h"
#include "sparkinfer/kernels/moe.h"

#include <memory>
#include <cstdio>

namespace sparkinfer {
namespace moe {

struct Router::Impl {
    RouterConfig cfg;
};

Router::Router(const RouterConfig& cfg) : impl_(new Impl{cfg}) {}
Router::~Router() = default;

void Router::route(const float* router_logits, int* expert_ids, float* expert_weights,
                   int* tokens_per_expert, int num_tokens, cudaStream_t stream) {
    if (num_tokens <= 0) return;
    const int E = impl_->cfg.num_experts, K = impl_->cfg.top_k;
    if (E <= 0 || K <= 0 || K > E) {
        fprintf(stderr, "[moe] route: invalid config (num_experts=%d top_k=%d)\n", E, K);
        return;
    }
    kernels::launch_moe_router(
        router_logits, expert_ids, expert_weights, tokens_per_expert,
        num_tokens, E, K,
        impl_->cfg.normalize_weights ? 1 : 0, stream);
}

std::vector<int> Router::last_expert_counts() const {
    return {};   // sync-free: counts remain on device
}

}} // namespace sparkinfer::moe
