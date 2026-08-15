#pragma once

namespace sparkinfer { namespace kernels {

// sparkinfer-internal weight type ids, carried through the same `*_qtype` ints as ggml type ids
// (12 = Q4_K, 13 = Q5_K, 14 = Q6_K, 8 = Q8_0). Deliberately far above every ggml type id: a
// dispatch that does not explicitly know one of these simply misses it and falls through to a
// safe path, instead of silently reading a block of the wrong size. None of these ever appear
// in a GGUF file -- they are produced at model load and consumed by a matching vec-dot.
//
//   Q3_A -- 3.5 bits/weight. Q4_K's asymmetric per-32 scale AND min and Q4_K's exact 6-bit
//           scale/min packing, with the 4-bit quant plane replaced by a 3-bit one (a 2-bit
//           plane plus a 1-bit plane). 112 B per 256 weights against Q4_K's 144 B.
//           Produced by launch_ffn_requant_q3a (quant.h), consumed by si_vec_dot_q3_A
//           (moe/expert_ffn_q4k.cu).
//
// Lives in its own header because expert_ffn_q4k.cu includes moe.h from *inside*
// namespace sparkinfer::kernels, so anything declared there is not reachable by its
// unqualified name from that file's kernels.
static constexpr int SI_QTYPE_Q3A = 112;   // == the block size in bytes, per 256 weights

// Compressed-tensors FP8 (E4M3) kept native for decode GEMV. Payload is
//   [bf16 scale[N] | e4m3 W[N*K]]
// matching launch_ct_dequant_fp8's per-row scale: W_bf16[r,c] = bf16(float(e4m3)*float(scale[r])).
// 108 is unused as a ggml type id (ggml's 108 is not a weight format we ship).
static constexpr int SI_QTYPE_FP8 = 108;

// Compressed-tensors NVFP4 (E2M1, block 16) kept native. Payload is
//   [256 B header: f32 global_scale at byte 0 |
//    ue4m3 scale[N*(K/16)] |
//    packed u8[N*(K/2)]]
// The 256-byte header keeps the packed region 256-aligned for CUTLASS TMA.
// Dequant matches launch_ct_dequant_nvfp4:
//   W[r,c] = e2m1(nibble) * ue4m3(scale[r,c/16]) / global_scale.
// 109 is unused as a ggml type id.
static constexpr int SI_QTYPE_NVFP4 = 109;
static constexpr int SI_NVFP4_HDR = 256;

}} // namespace sparkinfer::kernels
