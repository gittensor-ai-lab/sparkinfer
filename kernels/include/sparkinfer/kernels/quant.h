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

// Fused GGUF dequant -> per-row symmetric int8 (for launch_prefill_gemm_i8):
// q[r,c] = round(v[r,c] / scale[r]), scale[r] = max_c|v[r,c]| / 127, with v the
// exact fp32 dequant. Skips the bf16 scratch round-trip of dequant + row-quantize.
// Q4_K/Q6_K only; returns false (nothing launched) for other types.
bool launch_gguf_dequant_rows_i8(int ggml_type, const void* src, signed char* q, float* scale,
                                 int rows, int cols, cudaStream_t stream = nullptr);

// Dual-tensor variant (same ggml_type). Falls back to two single launches if unsupported.
bool launch_gguf_dequant_rows_i8_pair(int ggml_type,
                                      const void* src0, signed char* q0, float* scale0,
                                      const void* src1, signed char* q1, float* scale1,
                                      int rows, int cols, cudaStream_t stream = nullptr);

// Live-expert gather (see dequant_rows_i8_fast.h). Returns false if unsupported.
bool launch_gguf_dequant_rows_i8_gather(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream = nullptr);

bool launch_gguf_dequant_rows_i8_gather_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream = nullptr);

bool launch_gguf_dequant_rows_i8_mask(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream = nullptr);

bool launch_gguf_dequant_rows_i8_mask_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream = nullptr);

// bf16 transposes used to relayout GGUF [out,in] -> our [in,out].
void launch_transpose_bf16(const void* src, void* dst, int rows, int cols,
                           cudaStream_t stream = nullptr);          // [rows,cols]->[cols,rows]
void launch_transpose3d_bf16(const void* src, void* dst, int E, int A, int B,
                             cudaStream_t stream = nullptr);        // [E,A,B]->[E,B,A]

// Interleave two [n_heads*head_dim, in_dim] native-ggml (out,in) weight matrices into one
// [n_heads*2*head_dim, in_dim] matrix laid out per-head as [q_head | gate_head]: the layout
// split_q_gate_kernel (qwen36.cu) already expects for w.wq when Qwen35LayerWeights::q_has_gate
// is set. Used at load time when a GGUF ships q and gate as two separate tensors (e.g. Muse
// Glimmer's attn_q.weight/attn_gate.weight) rather than pre-fused into one (Qwen3.6's
// convention -- see qwen3_config_from_gguf's expect_dims on attn_q.weight for q_has_gate).
void launch_interleave_qgate_rows(const void* q, const void* gate, void* out,
                                  int n_heads, int head_dim, int in_dim,
                                  cudaStream_t stream = nullptr);

// Requantize a dense-FFN down projection bf16 -> Q4_K (ggml super-block layout consumed
// by si_vec_dot_q4_K). Load-time only; n_values must be a multiple of 256.
void launch_ffn_down_requant_q4k(const void* src_bf16, void* dst_q4k, long n_values,
                                 cudaStream_t stream = nullptr);

// Requantize a weight matrix bf16 -> Q3_A, sparkinfer's internal 3.5 bits/weight super-block
// (Q4_K's asymmetric per-32 scale and min and its 6-bit scale packing, with a 3-bit quant plane;
// 112 B per 256 weights against Q4_K's 144 B) consumed by si_vec_dot_q3_A. Load-time only; never
// read from or written to a GGUF. n_values must be a multiple of 256.
void launch_ffn_requant_q3a(const void* src_bf16, void* dst_q3a, long n_values,
                            cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
