// Coalesced Q4_K -> bf16 dequantization.
//
// See kernels/csrc/cuda/quant/dequant_gguf_fast.cu for the design notes. deq_q4k_kernel assigns one
// thread per 256-value super-block and scalar-stores all 256 bf16 outputs from it, so consecutive
// lanes write 512 B apart and every store burns a 32-byte sector to deliver 2 bytes (~16x write
// amplification, ~7% of DRAM peak). On an RTX 5090 that is 19.05 ms of GGUF load plus 4.42 ms of
// every Qwythos prefill (the 48 ssm_alpha/ssm_beta projections that #464's fused Q4K->int8 path
// leaves behind, since it only takes n_out >= 128). This translation unit runs the same math with
// one warp per super-block and one 16-byte coalesced store per lane.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Coalesced Q4_K dequant. Signature mirrors `launch_gguf_dequant` so it drops in as a one-line
// guard at the top of that launcher:
//
//     if (launch_gguf_dequant_fast(ggml_type, src, dst_bf16, n_values, stream)) return;
//
// Returns true if a kernel was launched, false if the caller should run its own dequant (not
// Q4_K, ragged n_values, or disabled by env).
//
// Output is BYTE-EXACT with deq_q4k_kernel — the per-value arithmetic and rounding are unchanged
// and only the thread->output mapping differs. This is load-bearing: this path is shared with
// Qwen3.6, and the bidir eval rejects any PR that moves the guard model's numerics.
//
// Env knobs:
//   SPARKINFER_DEQUANT_COALESCED  (default 1)  0 disables (A/B) -> falls through to deq_q4k_kernel.
bool launch_gguf_dequant_fast(int ggml_type, const void* src, void* dst_bf16, long n_values,
                              cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
