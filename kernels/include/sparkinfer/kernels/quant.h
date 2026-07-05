#pragma once
#include <cuda_runtime.h>

// int8/uint8 spelled as signed/unsigned char so this header is safe to include
// in nvcc's device compilation pass (libstdc++ <cstdint> is not device-parseable).

namespace sparkinfer { namespace kernels {

// Per-tensor symmetric int8 quantize: out = round(in / scale), scale = max|in|/127.
// scale is computed on device and written to *scale (1 float).
void launch_quantize_i8(const void* in_bf16, signed char* out, float* scale, int n,
                        cudaStream_t stream = nullptr);

// Inverse: out = in * scale.
void launch_dequantize_i8(const signed char* in, const float* scale, void* out_bf16, int n,
                          cudaStream_t stream = nullptr);

// Symmetric int4 block dequant. Two 4-bit signed values are packed per byte
// (low nibble first). Each block of `block` values shares one bf16 scale.
//   packed:  [n/2] bytes,  scales: [n/block] bf16,  out: [n] bf16
void launch_dequant_int4_block(const unsigned char* packed, const void* scales_bf16,
                               void* out_bf16, int n, int block,
                               cudaStream_t stream = nullptr);

// GGUF block dequant -> bf16 (natural ggml order). ggml_type: 0=F32,1=F16,
// 8=Q8_0,12=Q4_K,14=Q6_K. Q4_K/Q6_K validated byte-exact vs the gguf reference.
void launch_gguf_dequant(int ggml_type, const void* src, void* dst_bf16, long n_values,
                         cudaStream_t stream = nullptr);

// Requantize a bf16 weight tensor to GGUF-native Q6_K (n_values multiple of 256; dst holds
// (n_values/256)*210 bytes). Used to lower the Q8_0 projection weights to Q6_K at load.
void launch_gguf_requant_q6k(const void* src_bf16, void* dst_q6k, long n_values,
                             cudaStream_t stream = nullptr);

// bf16 transposes used to relayout GGUF [out,in] -> our [in,out].
void launch_transpose_bf16(const void* src, void* dst, int rows, int cols,
                           cudaStream_t stream = nullptr);          // [rows,cols]->[cols,rows]
void launch_transpose3d_bf16(const void* src, void* dst, int E, int A, int B,
                             cudaStream_t stream = nullptr);        // [E,A,B]->[E,B,A]

}} // namespace sparkinfer::kernels
