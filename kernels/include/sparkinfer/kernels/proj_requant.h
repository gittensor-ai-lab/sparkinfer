#pragma once
#include <cuda_runtime.h>

// Load-time Q4_K requantizer for attention/GDN projection weights, fit by Lloyd-max
// coordinate descent (see kernels/csrc/cuda/quant/proj_q4k_lloyd.cu). Input is bf16;
// output is the ggml Q4_K super-block layout consumed by si_vec_dot_q4_K. Load-time
// only; n_values must be a multiple of 256.
namespace sparkinfer { namespace kernels {

void launch_proj_requant_q4k_lloyd(const void* src_bf16, void* dst_q4k, long n_values,
                                   cudaStream_t stream = nullptr);

// bf16 -> Q8_0 (ggml layout: fp16 d = amax/127, then 32 int8), rounded as ggml's
// quantize_row_q8_0 does. n_values must be a multiple of 32.
void launch_requant_q8_0(const void* src_bf16, void* dst_q8_0, long n_values,
                         cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
