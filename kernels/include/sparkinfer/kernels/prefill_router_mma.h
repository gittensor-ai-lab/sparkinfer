#pragma once
// Tensor-core (bf16 wmma, fp32 accumulate) router-logits GEMM for the Qwen3.6
// MoE batched prefill: logits[N,E] = x[N,H] . W[E,H]^T. Same bf16 inputs and
// fp32 accumulation as the warp-dot reference in prefill_moe.cu -- only the
// fp32 summation order differs. See prefill_router_mma.cu for the rationale
// and measured numbers.
#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Returns false (caller falls back to launch_pfm_router_logits) when disabled
// via SPARKINFER_PREFILL_ROUTER_MMA=0 or the shape is not tile-aligned
// (E % 128, H % 32, or N < 128).
bool launch_pfm_router_logits_mma(const void* x, const void* W, float* logits,
                                  int n_tokens, int n_experts, int H,
                                  cudaStream_t stream);

}  // namespace kernels
}  // namespace sparkinfer
