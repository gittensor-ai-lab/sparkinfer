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
        for (int m = 0; m < 5; ++m) { trits[m] = static_cast<int8_t>(x % 3 - 1); x /= 3; }
        uint8_t block[kPtq1BlockBytes];
        ptq1_pack_block(trits, 1.0f, block);
        wide.insert(block[0]);
    }
    std::memset(trits, 0, sizeof(trits));
    for (int v = 0; v < 81; ++v) {
        int x = v;
        for (int m = 0; m < 4; ++m) {
            trits[kPtq1Wide * 5 + m] = static_cast<int8_t>(x % 3 - 1);
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
    test_carrier_alphabets_match_the_checkpoint();
    test_block_geometry();
    if (failures) { std::printf("ternary_ptq1_cpu_test: %d FAILURES\n", failures); return 1; }
    std::printf("ternary_ptq1_cpu_test: OK\n");
    return 0;
}
