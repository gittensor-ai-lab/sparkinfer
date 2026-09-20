#pragma once
// PTQ1_0 -- the ternary weight format of prism-ml/Ternary-Bonsai-2-27B (ggml type 143).
//
// Each weight is a trit {-1, 0, +1} with one FP16 scale per group of 128, so a block is 28 bytes
// and the format costs exactly 1.75 bits per weight. The trits are NOT stored as plain base-3
// digits: like ggml's TQ1_0, each byte holds a base-3 number scaled into the full 0..255 range, so
// a byte takes 243 distinct values for a 5-trit carrier and 81 for a 4-trit one, and a digit is
// recovered by multiplying and taking the high bits. Verified against the checkpoint itself: bytes
// 0..23 show 243 distinct values, bytes 24..25 show 81, and nothing else fits 28 bytes / 128
// weights.
//
//   bytes  0..23  5 trits each  (120)
//   bytes 24..25  4 trits each  (8)   -> 128 weights
//   bytes 26..27  FP16 group scale
//
// The checkpoint's weights are Hadamard-rotated offline (see prism.hadamard.* in the GGUF
// metadata); this header is only about unpacking, and says nothing about that rotation.
#include <cstddef>
#include <cstdint>

namespace sparkinfer {

inline constexpr int   kPtq1GgmlType   = 143;
inline constexpr int   kPtq1BlockElems = 128;
inline constexpr int   kPtq1BlockBytes = 28;
inline constexpr int   kPtq1Wide       = 24;   // bytes carrying 5 trits
inline constexpr int   kPtq1Narrow     = 2;    // bytes carrying 4 trits

// One 28-byte block -> 128 floats. `out` must hold kPtq1BlockElems values.
void ptq1_dequant_block(const uint8_t* block, float* out);

// n_elems must be a multiple of kPtq1BlockElems (every tensor in the checkpoint is).
void ptq1_dequant(const uint8_t* data, size_t n_elems, float* out);

// The trits of one block, without the scale, as -1/0/+1. Used by the transcoder and the tests.
void ptq1_unpack_trits(const uint8_t* block, int8_t* trits);

// The block's FP16 scale as a float.
float ptq1_block_scale(const uint8_t* block);

// Packs 128 trits (-1/0/+1) and a scale into a 28-byte block, inverse of ptq1_unpack_trits.
// Only the tests need this -- the checkpoint is produced elsewhere -- but a decoder without its
// encoder is a decoder nobody can test.
void ptq1_pack_block(const int8_t* trits, float scale, uint8_t* block);

}  // namespace sparkinfer
