// The native PTQ1_0 GEMV against the host decoder that has already been validated byte-for-byte
// against the checkpoint this model was quantized from.
//
// The point of the kernel is to read the 28-byte blocks as they are stored rather than expand
// them, so what has to hold is that its trit extraction and the host's agree -- including the
// position-major carrier walk, which is the one part of this format that decodes every value
// correctly while putting them in the wrong places.
#include "sparkinfer/kernels/ternary.h"
#include "sparkinfer/ternary_ptq1.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

namespace {

uint16_t to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

void test_gemv_matches_the_host_decoder() {
    using namespace sparkinfer;
    const int rows = 37, k = 512;            // rows deliberately not a multiple of the warps/CTA
    const int blocks_per_row = k / kPtq1BlockElems;

    // Build a weight matrix by packing trits, the same path the format's own tests use.
    std::vector<uint8_t> w((size_t)rows * blocks_per_row * kPtq1BlockBytes);
    std::vector<int8_t> trits(kPtq1BlockElems);
    uint32_t rng = 12345u;
    auto next = [&] { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            for (int i = 0; i < kPtq1BlockElems; ++i) trits[i] = (int8_t)((int)(next() % 3) - 1);
            const float scale = 0.003f + (float)(next() % 100) * 0.0004f;
            ptq1_pack_block(trits.data(), scale,
                            w.data() + ((size_t)r * blocks_per_row + b) * kPtq1BlockBytes);
        }
    }

    std::vector<float> x(k);
    for (int i = 0; i < k; ++i) x[i] = std::sin(0.07f * (float)i) * 1.7f;

    // Reference: decode with the host decoder, then a plain dot product.
    std::vector<float> want(rows, 0.0f);
    std::vector<float> row_w(k);
    for (int r = 0; r < rows; ++r) {
        ptq1_dequant(w.data() + (size_t)r * blocks_per_row * kPtq1BlockBytes, (size_t)k,
                     row_w.data());
        double acc = 0.0;
        for (int i = 0; i < k; ++i) acc += (double)row_w[i] * from_bf16(to_bf16(x[i]));
        want[r] = (float)acc;
    }

    std::vector<uint16_t> hx(k);
    for (int i = 0; i < k; ++i) hx[i] = to_bf16(x[i]);
    void* dx = nullptr;
    void* dw = nullptr;
    void* dy = nullptr;
    cudaMalloc(&dx, hx.size() * 2);
    cudaMalloc(&dw, w.size());
    cudaMalloc(&dy, (size_t)rows * 2);
    cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    kernels::launch_gemv_ptq1(dx, dw, dy, rows, k, 0);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint16_t> hy(rows);
    cudaMemcpy(hy.data(), dy, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);

    double worst = 0.0, scale = 0.0;
    for (int r = 0; r < rows; ++r) {
        worst = std::fmax(worst, std::fabs(from_bf16(hy[r]) - want[r]));
        scale = std::fmax(scale, std::fabs(want[r]));
    }
    // The kernel accumulates in float and rounds once to bf16, so agreement is to that rounding.
    CHECK(worst < 0.01 * scale);
    if (!(worst < 0.01 * scale)) std::printf("  worst %.6f against |y|max %.6f\n", worst, scale);
}

void test_a_single_trit_lands_where_the_host_puts_it() {
    // Narrower than the statistical check above: one weight set, everything else zero. If the
    // kernel's carrier walk disagrees with the host's, this puts the weight on the wrong input.
    using namespace sparkinfer;
    const int k = kPtq1BlockElems;
    for (int probe : {0, 3, 16, 79, 80, 95, 119, 120, 127}) {
        std::vector<int8_t> trits(k, 0);
        trits[probe] = 1;
        std::vector<uint8_t> w(kPtq1BlockBytes);
        ptq1_pack_block(trits.data(), 1.0f, w.data());

        std::vector<uint16_t> hx(k, to_bf16(0.0f));
        for (int i = 0; i < k; ++i) hx[i] = to_bf16((float)(i + 1));   // x[i] = i+1, so y == probe+1

        void* dx = nullptr;
        void* dw = nullptr;
        void* dy = nullptr;
        cudaMalloc(&dx, hx.size() * 2);
        cudaMalloc(&dw, w.size());
        cudaMalloc(&dy, 2);
        cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
        kernels::launch_gemv_ptq1(dx, dw, dy, 1, k, 0);
        cudaDeviceSynchronize();
        uint16_t hy = 0;
        cudaMemcpy(&hy, dy, 2, cudaMemcpyDeviceToHost);
        cudaFree(dx);
        cudaFree(dw);
        cudaFree(dy);

        CHECK(std::fabs(from_bf16(hy) - (float)(probe + 1)) < 1.0f);
        if (std::fabs(from_bf16(hy) - (float)(probe + 1)) >= 1.0f)
            std::printf("  probe %d -> %.2f (wanted %d)\n", probe, from_bf16(hy), probe + 1);
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("gemv_ptq1_gpu_test: SKIPPED (no CUDA device)\n");
        return 0;
    }
    test_gemv_matches_the_host_decoder();
    test_a_single_trit_lands_where_the_host_puts_it();
    std::printf("gemv_ptq1_gpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
