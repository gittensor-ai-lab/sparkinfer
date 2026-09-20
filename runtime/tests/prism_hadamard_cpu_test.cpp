// CPU-only test for the Ternary-Bonsai-2 Hadamard rotation (prism.hadamard.*).
//
// The point of the rotation is that W_rot . (H . diag(s) . x) == W . x, so the test checks that
// identity directly on random matrices -- if the runtime ever applied the sign after the transform,
// or skipped the 1/sqrt(block) normalisation, this is what would catch it.
#include "sparkinfer/prism_hadamard.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;
#define CHECK(expr)                                                       \
    do {                                                                  \
        if (!(expr)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

using namespace sparkinfer;

void test_transform_is_its_own_inverse() {
    const long block = 1024, n = block * 3;
    std::vector<float> x(n), original;
    for (long i = 0; i < n; ++i) x[i] = std::sin(0.37f * (float)i) * 3.0f;
    original = x;
    hadamard_transform(x.data(), n, block);
    bool moved = false;
    for (long i = 0; i < n; ++i) moved = moved || std::fabs(x[i] - original[i]) > 1e-4f;
    CHECK(moved);   // a no-op transform would pass the round trip below trivially
    hadamard_transform(x.data(), n, block);
    for (long i = 0; i < n; ++i) CHECK(std::fabs(x[i] - original[i]) < 1e-3f);
}

void test_transform_preserves_norm() {
    const long block = 256, n = block * 2;
    std::vector<float> x(n);
    double before = 0.0;
    for (long i = 0; i < n; ++i) { x[i] = (float)((i * 37 % 19) - 9); before += (double)x[i] * x[i]; }
    hadamard_transform(x.data(), n, block);
    double after = 0.0;
    for (long i = 0; i < n; ++i) after += (double)x[i] * x[i];
    CHECK(std::fabs(before - after) / before < 1e-5);
}

void test_blocks_are_independent() {
    // Blockwise means blockwise: touching one block must not move another.
    const long block = 8, n = 16;
    std::vector<float> a(n, 0.0f), b(n, 0.0f);
    a[0] = 1.0f; b[0] = 1.0f; b[block] = 5.0f;
    hadamard_transform(a.data(), n, block);
    hadamard_transform(b.data(), n, block);
    for (long i = 0; i < block; ++i) CHECK(std::fabs(a[i] - b[i]) < 1e-6f);
}

void test_known_two_point_case() {
    // H2 / sqrt(2) on (1, 0) is (1, 1)/sqrt(2): the smallest case worth writing down.
    std::vector<float> x{1.0f, 0.0f};
    hadamard_transform(x.data(), 2, 2);
    const float inv = 1.0f / std::sqrt(2.0f);
    CHECK(std::fabs(x[0] - inv) < 1e-6f);
    CHECK(std::fabs(x[1] - inv) < 1e-6f);
}

void test_rotated_weight_times_rotated_activation_reproduces_the_original() {
    // The identity the checkpoint relies on. W_rot = W . diag(s) . H is what a rotated weight
    // holds; feeding it H . diag(s) . x must give back W . x.
    const long in = 64, out = 5, block = 16;
    std::vector<int8_t> sign(in);
    std::vector<float> W((size_t)out * in), x(in);
    std::srand(7);
    for (long i = 0; i < in; ++i) {
        sign[i] = (std::rand() & 1) ? 1 : -1;
        x[i] = (float)(std::rand() % 2000 - 1000) / 250.0f;
    }
    for (size_t i = 0; i < W.size(); ++i) W[i] = (float)(std::rand() % 2000 - 1000) / 500.0f;

    std::vector<float> want(out, 0.0f);
    for (long r = 0; r < out; ++r)
        for (long c = 0; c < in; ++c) want[r] += W[(size_t)r * in + c] * x[c];

    // Rotate each row of W the way the producer does, along the input axis.
    std::vector<float> Wrot = W;
    for (long r = 0; r < out; ++r) {
        float* row = Wrot.data() + (size_t)r * in;
        hadamard_apply_sign(row, in, sign.data());
        hadamard_transform(row, in, block);
    }
    std::vector<float> xr = x;
    hadamard_rotate_activation(xr.data(), in, block, sign.data());

    for (long r = 0; r < out; ++r) {
        float got = 0.0f;
        for (long c = 0; c < in; ++c) got += Wrot[(size_t)r * in + c] * xr[c];
        CHECK(std::fabs(got - want[r]) < 1e-3f);
    }
}

void test_odd_sizes_are_left_alone() {
    // n not a multiple of the block would mean a partial rotation; refuse rather than corrupt.
    std::vector<float> x{1.0f, 2.0f, 3.0f};
    const std::vector<float> before = x;
    hadamard_transform(x.data(), 3, 2);
    for (size_t i = 0; i < x.size(); ++i) CHECK(x[i] == before[i]);
}

}  // namespace


void test_unrotate_undoes_rotate() {
    // The embedding path leans on this exactly: a row stored as R.e must come back as e.
    const long n = 2048, block = 1024;
    std::vector<int8_t> sign(n);
    for (long i = 0; i < n; ++i) sign[i] = (i * 7 + 3) % 5 < 2 ? -1 : 1;
    std::vector<float> x(n), original(n);
    for (long i = 0; i < n; ++i) x[i] = original[i] = std::sin(0.37f * (float)i) * (float)(1 + i % 11);

    sparkinfer::hadamard_rotate_activation(x.data(), n, block, sign.data());
    bool moved = false;
    for (long i = 0; i < n; ++i) moved = moved || std::fabs(x[i] - original[i]) > 1e-3f;
    CHECK(moved);   // guards against a rotation that quietly does nothing

    sparkinfer::hadamard_unrotate_activation(x.data(), n, block, sign.data());
    for (long i = 0; i < n; ++i) CHECK(std::fabs(x[i] - original[i]) < 1e-3f);
}

int main() {
    test_transform_is_its_own_inverse();
    test_transform_preserves_norm();
    test_blocks_are_independent();
    test_known_two_point_case();
    test_rotated_weight_times_rotated_activation_reproduces_the_original();
    test_unrotate_undoes_rotate();
    test_odd_sizes_are_left_alone();
    if (failures) { std::printf("prism_hadamard_cpu_test: %d FAILURES\n", failures); return 1; }
    std::printf("prism_hadamard_cpu_test: OK\n");
    return 0;
}
