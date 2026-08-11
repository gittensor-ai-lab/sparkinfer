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

}} // namespace sparkinfer::kernels
