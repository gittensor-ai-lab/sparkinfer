// CPU-only test for the PTQ1_0 ternary unpacker (ggml type 143, prism-ml Ternary-Bonsai-2).
//
// The format was derived from the checkpoint rather than from a spec, so the test pins the two
// things that derivation rests on: a block round-trips through pack/unpack, and the encoding is
// the SCALED base-3 one (a 5-trit carrier takes exactly 243 distinct byte values, a 4-trit one 81)
// rather than plain base-3 digits. A plain-base-3 decoder passes neither.
#include "sparkinfer/ternary_ptq1.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

namespace {

int failures = 0;
#define CHECK(expr)                                                                   \
    do {                                                                              \
        if (!(expr)) {                                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);               \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

using namespace sparkinfer;

void test_round_trip() {
    int8_t trits[kPtq1BlockElems];
    for (int i = 0; i < kPtq1BlockElems; ++i) trits[i] = static_cast<int8_t>((i * 7 + i / 5) % 3 - 1);
    uint8_t block[kPtq1BlockBytes];
    ptq1_pack_block(trits, 0.017685f, block);

    int8_t back[kPtq1BlockElems];
    ptq1_unpack_trits(block, back);
    for (int i = 0; i < kPtq1BlockElems; ++i) CHECK(back[i] == trits[i]);

    // The scale survives as FP16, and dequant is trit * scale.
    const float scale = ptq1_block_scale(block);
    CHECK(std::fabs(scale - 0.017685f) < 1e-4f);
    float out[kPtq1BlockElems];
    ptq1_dequant_block(block, out);
    for (int i = 0; i < kPtq1BlockElems; ++i)
        CHECK(std::fabs(out[i] - static_cast<float>(trits[i]) * scale) < 1e-6f);
}

void test_every_trit_pattern_survives() {
    // Every digit of every carrier, independently: a packing that drops or aliases one position
    // (the failure mode of guessing the layout) shows up here and nowhere else.
    for (int pos = 0; pos < kPtq1BlockElems; ++pos) {
        for (int v = -1; v <= 1; ++v) {
            int8_t trits[kPtq1BlockElems];
            std::memset(trits, 0, sizeof(trits));
            trits[pos] = static_cast<int8_t>(v);
            uint8_t block[kPtq1BlockBytes];
            ptq1_pack_block(trits, 1.0f, block);
            int8_t back[kPtq1BlockElems];
            ptq1_unpack_trits(block, back);
            for (int i = 0; i < kPtq1BlockElems; ++i) CHECK(back[i] == trits[i]);
        }
    }
}

void test_trits_come_out_in_ggml_tq1_0_order() {
    // The round-trip tests cannot catch a wrong trit ORDER, because they pack with whatever order
    // they unpack. This one states the layout outright: carrier byte j of a run owns the weights
    // at j, j+run, j+2*run, ... Reading it carrier-major instead permutes the weights inside every
    // group of 128 -- every value correct, every one in the wrong place, which is indistinguishable
    // from a bad rotation until you compare against the un-quantized checkpoint.
    uint8_t block[kPtq1BlockBytes], zero_block[kPtq1BlockBytes];
    int8_t trits[kPtq1BlockElems];
    std::memset(trits, 0, sizeof(trits));
    ptq1_pack_block(trits, 1.0f, zero_block);   // a zero trit is digit 1, so this is not all-zero

    // A single carrier byte with trit +1 at position m and 0 elsewhere: encode_carrier's value for
    // that is (3^m * 2) in base 3 terms, but we only care WHERE the +1 lands after unpacking.
    for (int m = 0; m < 5; ++m) {
        std::memset(trits, 0, sizeof(trits));
        trits[m * kPtq1WideRuns[0] + 3] = 1;          // byte 3 of the first run, position m
        ptq1_pack_block(trits, 1.0f, block);
        int8_t back[kPtq1BlockElems];
        ptq1_unpack_trits(block, back);
        CHECK(back[m * kPtq1WideRuns[0] + 3] == 1);
        for (int i = 0; i < kPtq1BlockElems; ++i)
            if (i != m * kPtq1WideRuns[0] + 3) CHECK(back[i] == 0);
        // and it must live in byte 3, not somewhere the carrier-major reading would put it
        CHECK(block[3] != zero_block[3]);
        for (int j = 0; j < kPtq1Wide + kPtq1Narrow; ++j)
            if (j != 3) CHECK(block[j] == zero_block[j]);
    }
    // second run starts at weight 80, and the 4-trit carriers at 120
    std::memset(trits, 0, sizeof(trits));
    trits[80 + 2 * kPtq1WideRuns[1] + 5] = 1;         // byte 5 of the 8-byte run, position 2
    ptq1_pack_block(trits, 1.0f, block);
    CHECK(block[kPtq1WideRuns[0] + 5] != zero_block[kPtq1WideRuns[0] + 5]);
    std::memset(trits, 0, sizeof(trits));
    trits[120 + 3 * kPtq1Narrow + 1] = 1;             // second 4-trit carrier, position 3
    ptq1_pack_block(trits, 1.0f, block);
    CHECK(block[kPtq1Wide + 1] != zero_block[kPtq1Wide + 1]);
}

void test_carrier_alphabets_match_the_checkpoint() {
    // What identified the format in the file: byte positions 0..23 take 243 distinct values across
    // real blocks and positions 24..25 take 81. Enumerating every trit combination must reproduce
    // exactly those alphabets -- and 243 != 256 is what rules out plain base-3 packing, whose
    // bytes would never exceed 242.
    std::set<int> wide, narrow;
    int8_t trits[kPtq1BlockElems];
    std::memset(trits, 0, sizeof(trits));
    for (int v = 0; v < 243; ++v) {
        int x = v;
        // carrier byte 0 of the 16-byte run owns weights 0, 16, 32, 48, 64 -- position-major
        for (int m = 0; m < 5; ++m) {
            trits[m * kPtq1WideRuns[0]] = static_cast<int8_t>(x % 3 - 1);
            x /= 3;
        }
        uint8_t block[kPtq1BlockBytes];
        ptq1_pack_block(trits, 1.0f, block);
        wide.insert(block[0]);
    }
    std::memset(trits, 0, sizeof(trits));
    for (int v = 0; v < 81; ++v) {
        int x = v;
        // the first 4-trit carrier owns weights 120, 122, 124, 126
        for (int m = 0; m < 4; ++m) {
            trits[kPtq1Wide * 5 + m * kPtq1Narrow] = static_cast<int8_t>(x % 3 - 1);
            x /= 3;
        }
        uint8_t block[kPtq1BlockBytes];
        ptq1_pack_block(trits, 1.0f, block);
        narrow.insert(block[kPtq1Wide]);
    }
    CHECK(wide.size() == 243);
    CHECK(narrow.size() == 81);
    CHECK(*wide.rbegin() > 242);      // the scaled encoding reaches past plain base-3's 242
    CHECK(*narrow.rbegin() > 80);
}

// A Q4_K decoder, written here rather than borrowed, so the transcoder is checked against the
// format's own definition instead of against itself: value = d*sc[j]*q - dmin*m[j], eight 32-wide
// sub-blocks, six-bit scale pairs packed the way ggml packs them.
void q4k_dequant_block(const uint8_t* blk, float* out) {
    uint16_t hd, hm;
    std::memcpy(&hd, blk, 2);
    std::memcpy(&hm, blk + 2, 2);
    auto half_to_float = [](uint16_t h) {
        const uint32_t sign = (h & 0x8000u) << 16;
        const uint32_t exp = (h >> 10) & 0x1Fu;
        const uint32_t man = h & 0x3FFu;
        uint32_t bits = exp == 0 ? sign : (sign | ((exp + 127 - 15) << 23) | (man << 13));
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    };
    const float d = half_to_float(hd), dmin = half_to_float(hm);
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    for (int j = 0; j < 8; ++j) {
        uint8_t sc, mn;
        if (j < 4) { sc = scales[j] & 63; mn = scales[j + 4] & 63; }
        else {
            sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
            mn = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
        const int half = j / 2;
        const int lo = (j % 2) == 0;
        for (int i = 0; i < 32; ++i) {
            const uint8_t byte = qs[half * 32 + i];
            const int q = lo ? (byte & 0xF) : (byte >> 4);
            out[j * 32 + i] = d * (float)sc * (float)q - dmin * (float)mn;
        }
    }
}

void test_q4k_transcode_keeps_the_ternary_grid() {
    // Two PTQ1_0 groups with different scales become one Q4_K superblock. Every value must come
    // back on the {-s, 0, +s} grid of ITS group, within the error of Q4_K's six-bit scale grid.
    int8_t trits[256];
    for (int i = 0; i < 256; ++i) trits[i] = static_cast<int8_t>((i * 5 + i / 7) % 3 - 1);
    const float s0 = 0.018967f, s1 = 0.014221f;
    uint8_t src[2 * kPtq1BlockBytes];
    ptq1_pack_block(trits, s0, src);
    ptq1_pack_block(trits + 128, s1, src + kPtq1BlockBytes);

    uint8_t q4k[kQ4KBlockBytes];
    ptq1_to_q4k(src, 256, q4k);
    float got[256];
    q4k_dequant_block(q4k, got);

    float worst = 0.0f;
    for (int i = 0; i < 256; ++i) {
        const float s = i < 128 ? s0 : s1;
        const float want = static_cast<float>(trits[i]) * s;
        const float err = std::fabs(got[i] - want) / s;   // relative to the group's own scale
        if (err > worst) worst = err;
    }
    CHECK(worst < 0.01f);   // neighbouring groups with similar scales: the grid is fine
    if (worst >= 0.01f) std::printf("  worst relative error %.4f\n", worst);
}

void test_q4k_transcode_survives_scales_that_push_d_subnormal() {
    // Q4_K's d is 1/63 of the weights it describes, so a group scale under ~3.8e-3 -- ordinary in
    // this checkpoint -- drives d below fp16's smallest normal (2^-14). A writer that flushes there
    // stores d = 0 and the whole superblock decodes to a constant -dmin*m: eight trit steps of
    // error on weights that should be exactly zero. Scales here bracket that cliff.
    int8_t trits[256];
    for (int i = 0; i < 256; ++i) trits[i] = static_cast<int8_t>((i % 3) - 1);
    const float scales[] = {3.6e-3f, 1.0e-3f, 2.0e-4f, 5.0e-5f};
    for (float s0 : scales) {
        for (float s1 : scales) {
            uint8_t src[2 * kPtq1BlockBytes];
            ptq1_pack_block(trits, s0, src);
            ptq1_pack_block(trits + 128, s1, src + kPtq1BlockBytes);
            uint8_t q4k[kQ4KBlockBytes];
            ptq1_to_q4k(src, 256, q4k);
            float got[256];
            q4k_dequant_block(q4k, got);
            for (int i = 0; i < 256; ++i) {
                const float s = i < 128 ? s0 : s1;
                if (trits[i] == 0) CHECK(got[i] == 0.0f);          // exact, not merely small
                else CHECK(std::fabs(got[i] - trits[i] * s) < 0.6f * s);
            }
        }
    }
}

void test_q4k_transcode_degrades_when_group_scales_diverge() {
    // The limitation this format has, pinned so nobody trusts the transcode further than it goes.
    // A Q4_K superblock spans TWO ternary groups that share one six-bit scale ladder, so when their
    // scales diverge the smaller group lands on a coarse rung. Real tensors sit at 0.34% RMS, but
    // that is because adjacent groups there happen to be close; a ten-to-one pair is several
    // percent of a trit step. A native ternary kernel keeps every group its own scale.
    int8_t trits[256];
    for (int i = 0; i < 256; ++i) trits[i] = static_cast<int8_t>((i % 3) - 1);
    const float s0 = 0.02f, s1 = 0.002f;        // a ten-to-one spread, as seen in the checkpoint
    uint8_t src[2 * kPtq1BlockBytes];
    ptq1_pack_block(trits, s0, src);
    ptq1_pack_block(trits + 128, s1, src + kPtq1BlockBytes);
    uint8_t q4k[kQ4KBlockBytes];
    ptq1_to_q4k(src, 256, q4k);
    float got[256];
    q4k_dequant_block(q4k, got);

    float worst_small = 0.0f;
    for (int i = 128; i < 256; ++i) {
        const float want = static_cast<float>(trits[i]) * s1;
        worst_small = std::fmax(worst_small, std::fabs(got[i] - want) / s1);
    }
    CHECK(worst_small > 0.02f);    // it really does degrade -- this is documentation, not a wish
    CHECK(worst_small < 0.30f);    // ...but stays bounded, so output is degraded rather than noise
}

void test_block_geometry() {
    // 24 carriers of 5 trits plus 2 of 4 is the only split that fills 128 weights in 26 payload
    // bytes; the last two bytes are the FP16 scale, giving 1.75 bits per weight.
    CHECK(kPtq1Wide * 5 + kPtq1Narrow * 4 == kPtq1BlockElems);
    CHECK(kPtq1Wide + kPtq1Narrow + 2 == kPtq1BlockBytes);
    CHECK(kPtq1BlockBytes * 8.0 / kPtq1BlockElems == 1.75);
    CHECK(kPtq1GgmlType == 143);
}

}  // namespace

int main() {
    test_round_trip();
    test_every_trit_pattern_survives();
    test_trits_come_out_in_ggml_tq1_0_order();
    test_carrier_alphabets_match_the_checkpoint();
    test_block_geometry();
    test_q4k_transcode_keeps_the_ternary_grid();
    test_q4k_transcode_survives_scales_that_push_d_subnormal();
    test_q4k_transcode_degrades_when_group_scales_diverge();
    if (failures) { std::printf("ternary_ptq1_cpu_test: %d FAILURES\n", failures); return 1; }
    std::printf("ternary_ptq1_cpu_test: OK\n");
    return 0;
}
