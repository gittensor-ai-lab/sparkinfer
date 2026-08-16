#pragma once
#include <cuda_runtime.h>

// HuggingFace "compressed-tensors" checkpoints (NVIDIA ModelOpt / llm-compressor --
// e.g. unsloth/Qwen3.8-27B-NVFP4). Two consumers:
//   - load-time dequant to bf16, then the same Q4_K requant pipeline load_gguf() uses (FP8 FFN)
//   - decode GEMV that keeps checkpoint FP8 native (SI_QTYPE_FP8 packed payload; launch_gemv_fp8)
//   - decode GEMV / prefill GEMM that keep checkpoint NVFP4 native (SI_QTYPE_NVFP4; launch_gemv_nvfp4)

namespace sparkinfer { namespace kernels {

// FP8 (E4M3) weight, per-output-channel (row) scale: out[r,c] = float(e4m3(w[r,c])) * float(scale[r])
// w: [rows,cols] raw e4m3 bytes, scale: [rows] bf16, out: [rows,cols] bf16.
void launch_ct_dequant_fp8(const void* w_e4m3, const void* scale_bf16, void* out_bf16,
                           int rows, int cols, cudaStream_t stream = nullptr);

// Same dequant, packed as [bf16 scale[rows] | e4m3 w[rows*cols]] (SI_QTYPE_FP8 decode payload).
void launch_ct_dequant_fp8_packed(const void* packed, void* out_bf16,
                                  int rows, int cols, cudaStream_t stream = nullptr);

// NVFP4 (E2M1), block_size=16 group scale (UE4M3) + a single tensor-wide F32 global scale:
//   out[r,c] = float(e2m1(packed nibble)) * float(ue4m3(group_scale[r, c/16])) * global_scale
// packed: [rows, cols/2] U8 (2 nibbles/byte, low nibble = even column), group_scale: [rows,
// cols/16] raw ue4m3 bytes (note: tagged F8_E4M3 in the safetensors header, but interpreted as
// CUTLASS's unsigned e4m3 -- the standard NVIDIA NVFP4 export convention; see the loader's own
// comment for why), out: [rows,cols] bf16. cols must be a multiple of 16.
void launch_ct_dequant_nvfp4(const void* packed_u8, const void* group_scale_ue4m3,
                             float global_scale, void* out_bf16, int rows, int cols,
                             cudaStream_t stream = nullptr);

// Same dequant, but the global scale is read from device memory. For callers holding an
// SI_QTYPE_NVFP4 payload (whose 256 B header stores the scale on the device) inside batched
// prefill: fetching it host-side would need a D2H copy, which under the CUDA graph capture that
// prefill runs in is illegal, not merely slow.
void launch_ct_dequant_nvfp4_dev(const void* packed_u8, const void* group_scale_ue4m3,
                                 const float* global_scale_dev, void* out_bf16, int rows, int cols,
                                 cudaStream_t stream = nullptr);

// Fused NVFP4 -> per-row int8 (q[rows,cols] + scale[rows]) for the int8 projection path, which
// otherwise dequantizes to a bf16 scratch and row-quantizes out of it -- a full bf16 write and
// re-read of the weight that nothing else ever looks at.
//
// Bit-identical to launch_ct_dequant_nvfp4_dev followed by launch_prefill_quantize_rows_i8: the
// same eight columns per thread, the same bf16 rounding, the same amax over that value set (a max,
// so associative and exact), the same d = amax/127.0f and roundf, the same amax==0 -> 0 rule.
//
// Returns false for shapes it does not handle (cols % 16, or a K needing more than four register
// slots) so the caller keeps the two-kernel path. SPARKINFER_CT_NVFP4_ROWS_I8=0 disables.
bool launch_ct_dequant_nvfp4_rows_i8(const void* packed_u8, const void* group_scale_ue4m3,
                                     const float* global_scale_dev, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream = nullptr);

// Exhaustive equivalence check for the per-group basis the NVFP4 checkpoint decode uses: every
// E2M1 nibble (16) against every group-scale byte (256), over a sweep of global scales, compared
// against the per-value `v * s / global_scale` it replaces. The encoding space is small enough to
// cover completely, so this proves the identity rather than sampling it.
//
// *bf16_mismatch is the one that must be zero: bf16 is what the decode stores. *fp32_finite_
// mismatch must also be zero; it excludes the two NaN scale encodings (0x7F/0xFF), where both
// results are NaN and differ only in the NaN's sign bit and which no real checkpoint contains.
// Returns false if the check could not run.
bool ct_nvfp4_basis_selftest(unsigned long long* fp32_finite_mismatch,
                             unsigned long long* bf16_mismatch, int* combinations = nullptr);

}} // namespace sparkinfer::kernels
