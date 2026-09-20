// The device Hadamard rotation against the host one the loader already uses.
//
// These two have to agree exactly, because they are the two halves of one decision: fold the
// rotation into the weights at load (host) or apply it to activations at runtime (device). A
// checkpoint loaded one way and served the other would be silently wrong in a way that still
// produces fluent text -- which is how long the equivalent mistake went unnoticed the first time.
#include "sparkinfer/kernels/hadamard.h"
#include "sparkinfer/prism_hadamard.h"

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

// Runs one direction on the device and returns the result as floats.
std::vector<float> on_device(const std::vector<float>& x, const std::vector<int8_t>& sign,
                             int width, int block, bool rotate) {
    std::vector<uint16_t> hx(x.size());
    for (size_t i = 0; i < x.size(); ++i) hx[i] = to_bf16(x[i]);

    void* dx = nullptr;
    void* dy = nullptr;
    signed char* ds = nullptr;
    cudaMalloc(&dx, hx.size() * 2);
    cudaMalloc(&dy, hx.size() * 2);
    cudaMalloc((void**)&ds, sign.size());
    cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sign.data(), sign.size(), cudaMemcpyHostToDevice);

    if (rotate)
        sparkinfer::kernels::launch_hadamard_rotate_bf16(dx, dy, ds, (long)x.size(), width, block, 0);
    else
        sparkinfer::kernels::launch_hadamard_unrotate_bf16(dx, dy, ds, (long)x.size(), width, block, 0);
    cudaDeviceSynchronize();

    std::vector<uint16_t> hy(hx.size());
    cudaMemcpy(hy.data(), dy, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(ds);

    std::vector<float> out(hy.size());
    for (size_t i = 0; i < hy.size(); ++i) out[i] = from_bf16(hy[i]);
    return out;
}

}  // namespace

void test_device_rotation_matches_the_host() {
    const int width = 5120, block = 1024, rows = 3;   // this checkpoint's residual width
    std::vector<int8_t> sign(width);
    for (int i = 0; i < width; ++i) sign[i] = (i * 7 + 3) % 5 < 2 ? -1 : 1;
    std::vector<float> x((size_t)rows * width);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(0.013f * (float)i) * (1.0f + (float)(i % 17) * 0.1f);

    for (bool rotate : {true, false}) {
        std::vector<float> want = x;
        for (int r = 0; r < rows; ++r) {
            float* row = want.data() + (size_t)r * width;
            if (rotate)
                sparkinfer::hadamard_rotate_activation(row, width, block, sign.data());
            else
                sparkinfer::hadamard_unrotate_activation(row, width, block, sign.data());
        }
        const std::vector<float> got = on_device(x, sign, width, block, rotate);

        double worst = 0.0, rms = 0.0;
        for (size_t i = 0; i < want.size(); ++i) {
            worst = std::fmax(worst, std::fabs(got[i] - want[i]));
            rms += (double)want[i] * want[i];
        }
        rms = std::sqrt(rms / (double)want.size());
        // bf16 carries ~3 decimal digits and the transform sums 1024 of them, so agreement is to
        // bf16 rounding rather than to the bit.
        CHECK(worst < 0.02 * rms * std::sqrt((double)block));
        if (!(worst < 0.02 * rms * std::sqrt((double)block)))
            std::printf("  %s: worst %.6f rms %.6f\n", rotate ? "rotate" : "unrotate", worst, rms);
    }
}

void test_the_two_directions_undo_each_other_on_device() {
    const int width = 6144, block = 1024;   // the GDN value width
    std::vector<int8_t> sign(width);
    for (int i = 0; i < width; ++i) sign[i] = (i % 3) == 0 ? -1 : 1;
    std::vector<float> x(width);
    for (int i = 0; i < width; ++i) x[i] = std::cos(0.021f * (float)i) * 2.0f;

    const std::vector<float> rotated = on_device(x, sign, width, block, true);
    bool moved = false;
    for (int i = 0; i < width; ++i) moved = moved || std::fabs(rotated[i] - x[i]) > 1e-2f;
    CHECK(moved);

    const std::vector<float> back = on_device(rotated, sign, width, block, false);
    double worst = 0.0;
    for (int i = 0; i < width; ++i) worst = std::fmax(worst, std::fabs(back[i] - x[i]));
    CHECK(worst < 0.1);
    if (worst >= 0.1) std::printf("  round trip worst %.6f\n", worst);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("hadamard_rotate_gpu_test: SKIPPED (no CUDA device)\n");
        return 0;
    }
    test_device_rotation_matches_the_host();
    test_the_two_directions_undo_each_other_on_device();
    std::printf("hadamard_rotate_gpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
