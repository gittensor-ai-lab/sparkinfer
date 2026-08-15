#pragma once
#include <cuda_runtime.h>

// HuggingFace "compressed-tensors" checkpoints (NVIDIA ModelOpt / llm-compressor --
// e.g. unsloth/Qwen3.8-27B-NVFP4). Two consumers:
//   - load-time dequant to bf16, then the same Q4_K / NVFP4 requant pipeline load_gguf() uses
//   - decode GEMV that keeps checkpoint FP8 native (SI_QTYPE_FP8 packed payload; launch_gemv_fp8)

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

}} // namespace sparkinfer::kernels
