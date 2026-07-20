#pragma once
// Load-time int8 row scales for the MoE expert stacks, and the matching per-pass quantizer.
//
// launch_gguf_dequant_rows_i8 (quant.h) fuses "GGUF row -> symmetric int8" by decoding each
// element twice: once for the row amax that fixes the scale, once to quantize against it. For a
// weight that is re-quantized on every pass — the 256 expert matrices of a Qwen3.6 MoE layer, redone
// per batched-prefill call — the amax half is repeated work, because scale[r] is a function of the
// static quantized weight alone.
//
// Split it: compute the scales once at load (launch_expert_row_scales_i8), then have the per-pass
// dequant consume them (launch_expert_rows_i8_scaled) and decode each element exactly once. The max
// reduction is exact and order-independent, so the scales — and the int8 rows derived from them —
// are bit-identical to what the fused kernel produces.
//
// Kept in its own translation unit rather than folded into dequant_gguf.cu: this is an expert-stack
// concern with a different lifetime (load vs per-pass) from that file's general dequant helpers.
//
// Q4_K (12), Q5_K (13) and Q6_K (14) only; both entry points return false, having launched nothing,
// for any other ggml type or a cols that is not a multiple of 256.

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// scale[r] = max_c |v[r,c]| / 127 over the exact fp32 dequant of row r. rows x cols elements.
bool launch_expert_row_scales_i8(int ggml_type, const void* src, float* scale,
                                 int rows, int cols, cudaStream_t stream = nullptr);

// q[r,c] = round(v[r,c] / scale[r]) using the precomputed scale (never written here).
bool launch_expert_rows_i8_scaled(int ggml_type, const void* src, signed char* q,
                                  const float* scale, int rows, int cols,
                                  cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace sparkinfer
