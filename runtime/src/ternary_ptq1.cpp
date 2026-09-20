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
    const int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = bits & 0x7FFFFFu;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        // Subnormal, or zero. Shifting the implicit leading one back in is what keeps a scale of
        // 5e-5 from becoming a scale of nothing; flushing here silently zeroed whole weight groups.
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t full = man | 0x800000u;
        const int shift = 14 - exp;                            // 14..24
        const uint32_t q = (full + (1u << (shift - 1)) - 1 + ((full >> shift) & 1u)) >> shift;
        return static_cast<uint16_t>(sign | q);
    }
    const uint32_t q = (man + 0x0FFFu + ((man >> 13) & 1u)) >> 13;   // round to nearest, ties to even
    return static_cast<uint16_t>(sign | ((exp << 10) + q));          // a mantissa carry bumps exp
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

void ptq1_to_q4k(const uint8_t* src, size_t n_elems, uint8_t* dst) {
    const size_t supers = n_elems / kQ4KBlockElems;
    for (size_t sb = 0; sb < supers; ++sb) {
        const uint8_t* b0 = src + (2 * sb) * kPtq1BlockBytes;
        const uint8_t* b1 = b0 + kPtq1BlockBytes;
        int8_t trits[kQ4KBlockElems];
        ptq1_unpack_trits(b0, trits);
        ptq1_unpack_trits(b1, trits + kPtq1BlockElems);
        const float s0 = ptq1_block_scale(b0), s1 = ptq1_block_scale(b1);

        // Q4_K stores value = d*sc[j]*q - dmin*m[j] over eight 32-element sub-blocks, with sc and m
        // six bits each. A ternary group needs d*sc = s and dmin*m = 8s, so pick d and dmin to put
        // both group scales as high on the 6-bit grid as they fit.
        const float smax = s0 > s1 ? s0 : s1;
        // d must stay a NORMAL fp16. smax/63 is 63x smaller than the weights it describes, and for
        // this checkpoint's scales (1e-4 .. 2e-2) that lands under fp16's smallest normal for any
        // group below ~3.8e-3 -- which is most of blk.31.ffn_down. Clamping costs rungs on the
        // six-bit ladder for those groups and costs nothing for the rest.
        const float kMinNormalFp16 = 6.103515625e-5f;            // 2^-14
        float d = smax / 63.0f;
        if (d < kMinNormalFp16) d = kMinNormalFp16;
        const uint16_t hd = float_to_fp16(d);
        d = fp16_to_float(hd);                                   // derive sc from the stored d, not the ideal one
        const float dmin = 8.0f * d;                             // exact: scaling by 8 only moves the exponent
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; ++j) {
            const float s = (j < 4) ? s0 : s1;
            const int q_sc = d > 0.0f ? (int)(s / d + 0.5f) : 0;
            sc[j] = (uint8_t)(q_sc < 0 ? 0 : (q_sc > 63 ? 63 : q_sc));
            // m is the SAME rung as sc, which is what makes a zero trit decode to exactly zero:
            // d*sc*8 - dmin*m = 8*d*(sc - m) = 0. Rounding the two independently loses that.
            mn[j] = sc[j];
        }

        uint8_t* out = dst + sb * kQ4KBlockBytes;
        const uint16_t hm = float_to_fp16(dmin);
        std::memcpy(out, &hd, 2);
        std::memcpy(out + 2, &hm, 2);
        uint8_t* scales = out + 4;
        // ggml's six-bit pairs: j<4 keeps sc and m whole, j>=4 splits them across two bytes.
        for (int j = 0; j < 4; ++j) { scales[j] = sc[j]; scales[j + 4] = mn[j]; }
        for (int j = 4; j < 8; ++j) {
            scales[j + 4] = (uint8_t)((sc[j] & 0xF) | ((mn[j] & 0xF) << 4));
            scales[j - 4] = (uint8_t)(scales[j - 4] | ((sc[j] >> 4) << 6));
            scales[j] = (uint8_t)(scales[j] | ((mn[j] >> 4) << 6));
        }
        uint8_t* qs = out + 16;
        // Nibbles are interleaved 32 apart within each 64-element half, as ggml packs them.
        for (int half = 0; half < 4; ++half) {
            const int8_t* t = trits + half * 64;
            uint8_t* q = qs + half * 32;
            for (int i = 0; i < 32; ++i)
                q[i] = (uint8_t)((t[i] + 8) | ((t[i + 32] + 8) << 4));
        }
    }
}

void ptq1_pack_block(const int8_t* trits, float scale, uint8_t* block) {
    for (int j = 0; j < kPtq1Wide; ++j) block[j] = encode_carrier(trits + j * 5, 5);
    for (int j = 0; j < kPtq1Narrow; ++j)
        block[kPtq1Wide + j] = encode_carrier(trits + kPtq1Wide * 5 + j * 4, 4);
    const uint16_t h = float_to_fp16(scale);
    std::memcpy(block + kPtq1BlockBytes - 2, &h, sizeof(h));
}

}  // namespace sparkinfer
