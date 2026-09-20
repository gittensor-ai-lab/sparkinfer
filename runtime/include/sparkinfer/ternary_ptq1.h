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

// Transcodes PTQ1_0 to Q4_K, the format this runtime's kernels already read. A ternary value sits
// exactly on Q4_K's grid -- value = d*sc*q - dmin*m with q in {7,8,9} gives {-s, 0, +s} -- so the
// only error is in representing the two group scales of a 256-element superblock on Q4_K's 6-bit
// scale grid: 0.34% RMS measured across real tensors, with zero trits exact and no weight off the
// ternary grid. The one sharp edge is that Q4_K's d is 1/63 of the scale it describes and so goes
// subnormal in fp16 for any group below ~3.8e-3; see the clamp in ptq1_to_q4k. Where the two
// groups of a superblock have very different scales the smaller one lands on a coarse rung, worst
// case a few percent of a trit step. That buys a model that runs on every existing kernel at 0.5625
// bytes per weight (~15 GB for 27B) while a native ternary kernel is written.
//
// `n_elems` must be a multiple of 256 (two PTQ1_0 blocks per Q4_K superblock). Writes
// n_elems/256 * 144 bytes.
void ptq1_to_q4k(const uint8_t* src, size_t n_elems, uint8_t* dst);

inline constexpr int kQ4KBlockElems = 256;
inline constexpr int kQ4KBlockBytes = 144;

// Packs 128 trits (-1/0/+1) and a scale into a 28-byte block, inverse of ptq1_unpack_trits.
// Only the tests need this -- the checkpoint is produced elsewhere -- but a decoder without its
// encoder is a decoder nobody can test.
void ptq1_pack_block(const int8_t* trits, float scale, uint8_t* block);

}  // namespace sparkinfer
