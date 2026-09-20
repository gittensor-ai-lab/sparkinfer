#include "sparkinfer/ternary_ptq1.h"

#include <cmath>
#include <cstring>

namespace sparkinfer {
namespace {

constexpr int kPow3[5] = {1, 3, 9, 27, 81};

// The scaled base-3 encoding: a carrier of `digits` trits stores v = sum(t_m * 3^m) as
// round(v * 256 / 3^digits), which spreads 3^digits values over the whole byte. Multiplying that
// byte by 3^m and taking the top bits recovers digit m -- the same trick ggml's TQ1_0 uses, and
// the reason payload bytes here reach 255 while taking only 243 (or 81) distinct values.
inline int digit(uint8_t byte, int m) {
    const uint8_t q = static_cast<uint8_t>(byte * kPow3[m]);
    return (static_cast<uint16_t>(q) * 3) >> 8;
}

inline uint8_t encode_carrier(const int8_t* trits, int digits) {
    // Digit 0 is the MOST significant: decoding multiplies the byte by 3^m and reads the top bits,
    // so the digit read at m=0 is the one that was accumulated first. Getting this backwards still
    // round-trips position 0 and fails everywhere else, which is what the per-position test catches.
    int v = 0, place = 1;
    for (int m = 0; m < digits; ++m) {
        v = v * 3 + (trits[m] + 1);
        place *= 3;
    }
    // Spread the 3^digits values over the whole byte, as ggml's TQ1_0 does.
    return static_cast<uint8_t>((v * 256 + (place - 1)) / place);
}

inline float fp16_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {   // subnormal: normalise it
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
            bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = bits & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);          // flush tiny scales to zero
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (exp << 10) | (man >> 13));
}

}  // namespace

void ptq1_unpack_trits(const uint8_t* block, int8_t* trits) {
    int k = 0;
    for (int j = 0; j < kPtq1Wide; ++j)
        for (int m = 0; m < 5; ++m) trits[k++] = static_cast<int8_t>(digit(block[j], m) - 1);
    for (int j = 0; j < kPtq1Narrow; ++j)
        for (int m = 0; m < 4; ++m)
            trits[k++] = static_cast<int8_t>(digit(block[kPtq1Wide + j], m) - 1);
}

float ptq1_block_scale(const uint8_t* block) {
    uint16_t h;
    std::memcpy(&h, block + kPtq1BlockBytes - 2, sizeof(h));
    return fp16_to_float(h);
}

void ptq1_dequant_block(const uint8_t* block, float* out) {
    int8_t trits[kPtq1BlockElems];
    ptq1_unpack_trits(block, trits);
    const float scale = ptq1_block_scale(block);
    for (int i = 0; i < kPtq1BlockElems; ++i) out[i] = static_cast<float>(trits[i]) * scale;
}

void ptq1_dequant(const uint8_t* data, size_t n_elems, float* out) {
    const size_t blocks = n_elems / kPtq1BlockElems;
    for (size_t b = 0; b < blocks; ++b)
        ptq1_dequant_block(data + b * kPtq1BlockBytes, out + b * kPtq1BlockElems);
}

void ptq1_pack_block(const int8_t* trits, float scale, uint8_t* block) {
    for (int j = 0; j < kPtq1Wide; ++j) block[j] = encode_carrier(trits + j * 5, 5);
    for (int j = 0; j < kPtq1Narrow; ++j)
        block[kPtq1Wide + j] = encode_carrier(trits + kPtq1Wide * 5 + j * 4, 4);
    const uint16_t h = float_to_fp16(scale);
    std::memcpy(block + kPtq1BlockBytes - 2, &h, sizeof(h));
}

}  // namespace sparkinfer
