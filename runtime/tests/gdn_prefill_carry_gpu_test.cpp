// GPU test for carrying the Gated-DeltaNet recurrent state into a batched prefill scan that does not
// start at position 0 (launch_prefill_gdn_scan's carry_in). The launcher runs the chunk-parallel
// scan at 128 tokens and up and the sequential scan below, so both are exercised:
//
//   chunk      a pass split at a chunk boundary (a multiple of 32) reproduces the unsplit pass bit
//              for bit -- given the incoming state, the chunk scan is local to its chunks;
//   sequential a pass split anywhere reproduces the unsplit sequential pass bit for bit;
//   mixed      a chunk pass then a short sequential pass (the shape of a resumed prefill) lands near
//              the unsplit chunk pass -- the two scans round differently, so this is reported;
//   control    the same split with carry_in = false does NOT reproduce the outputs, which is the
//              bug this guards against and proves the comparisons above can see it.
//
// Synthetic inputs shaped like Qwen3.8-27B (16 q/k heads, 48 v heads, head_dim 128): q and k
// L2-normalised per head, as the conv + norm hands them to the scan; decays in (0, 1).

#include "sparkinfer/kernels/prefill.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int QH = 16, VH = 48, HD = 128;

uint16_t to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000;
        const int shift = 14 - exp;
        uint16_t m = (uint16_t)(mant >> shift);
        if ((mant >> (shift - 1)) & 1) m++;
        return (uint16_t)(sign | m);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    uint16_t out = (uint16_t)(sign | (exp << 10) | (mant >> 13));
    if (mant & 0x1000) out++;
    return out;
}

float from_bf16(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            x = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

struct Host {
    int n = 0;
    std::vector<uint16_t> q, k, v, alpha, beta, dt, a;
};

Host make_inputs(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    Host h;
    h.n = n;
    auto unit_heads = [&](std::vector<uint16_t>& dst, int heads) {
        dst.resize((size_t)n * heads * HD);
        std::vector<float> row(HD);
        for (int t = 0; t < n; t++)
            for (int hh = 0; hh < heads; hh++) {
                double norm = 0;
                for (int d = 0; d < HD; d++) { row[d] = nd(rng); norm += row[d] * row[d]; }
                const float inv = (float)(1.0 / std::sqrt(norm));
                for (int d = 0; d < HD; d++) dst[((size_t)t * heads + hh) * HD + d] = to_bf16(row[d] * inv);
            }
    };
    unit_heads(h.q, QH);
    unit_heads(h.k, QH);
    h.v.resize((size_t)n * VH * HD);
    for (auto& x : h.v) x = to_bf16(0.5f * nd(rng));
    h.alpha.resize((size_t)n * VH);
    h.beta.resize((size_t)n * VH);
    for (auto& x : h.alpha) x = to_bf16(0.5f * nd(rng));
    for (auto& x : h.beta) x = to_bf16(nd(rng));
    h.dt.resize(VH);
    h.a.resize(VH);
    for (auto& x : h.dt) x = to_bf16(0.1f * nd(rng));
    for (auto& x : h.a) x = to_bf16(-(0.5f + 0.5f * std::fabs(nd(rng))));
    return h;
}

struct Dev {
    int n = 0;
    uint16_t* q = nullptr; uint16_t* k = nullptr; uint16_t* v = nullptr;
    uint16_t* alpha = nullptr; uint16_t* beta = nullptr; uint16_t* dt = nullptr; uint16_t* a = nullptr;
    float* state = nullptr;
    uint16_t* out = nullptr;
};

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

Dev to_device(const Host& h) {
    Dev d;
    d.n = h.n;
    d.q = upload(h.q); d.k = upload(h.k); d.v = upload(h.v);
    d.alpha = upload(h.alpha); d.beta = upload(h.beta); d.dt = upload(h.dt); d.a = upload(h.a);
    cudaMalloc(&d.state, (size_t)VH * HD * HD * sizeof(float));
    cudaMalloc(&d.out, (size_t)h.n * VH * HD * sizeof(uint16_t));
    return d;
}

void free_device(Dev& d) {
    for (void* p : {(void*)d.q, (void*)d.k, (void*)d.v, (void*)d.alpha, (void*)d.beta, (void*)d.dt,
                    (void*)d.a, (void*)d.state, (void*)d.out})
        cudaFree(p);
}

// [begin, end) of the inputs, continuing whatever is in d.state when carry.
void scan(const Dev& d, int begin, int end, bool carry, cudaStream_t st) {
    sparkinfer::kernels::launch_prefill_gdn_scan(
        d.q + (size_t)begin * QH * HD, d.k + (size_t)begin * QH * HD, d.v + (size_t)begin * VH * HD,
        d.alpha + (size_t)begin * VH, d.beta + (size_t)begin * VH, d.dt, d.a, d.state,
        d.out + (size_t)begin * VH * HD, end - begin, QH, VH, HD, /*qh_block=*/true, st, carry);
}

struct Run {
    std::vector<uint16_t> out;
    std::vector<float> state;
};

// Zero state, then the given passes in order; the first never carries.
Run run(const Dev& d, const std::vector<int>& bounds, bool carry_later, cudaStream_t st) {
    cudaMemset(d.state, 0, (size_t)VH * HD * HD * sizeof(float));
    cudaMemset(d.out, 0, (size_t)d.n * VH * HD * sizeof(uint16_t));
    for (size_t i = 0; i + 1 < bounds.size(); i++) scan(d, bounds[i], bounds[i + 1], i > 0 && carry_later, st);
    cudaStreamSynchronize(st);
    Run r;
    r.out.resize((size_t)d.n * VH * HD);
    r.state.resize((size_t)VH * HD * HD);
    cudaMemcpy(r.out.data(), d.out, r.out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.state.data(), d.state, r.state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return r;
}

struct Diff {
    bool identical = false;
    double out_max = 0, state_max = 0;
};

Diff diff(const Run& x, const Run& y) {
    Diff df;
    df.identical = x.out == y.out &&
                   std::memcmp(x.state.data(), y.state.data(), x.state.size() * sizeof(float)) == 0;
    for (size_t i = 0; i < x.out.size(); i++)
        df.out_max = std::max(df.out_max, (double)std::fabs(from_bf16(x.out[i]) - from_bf16(y.out[i])));
    for (size_t i = 0; i < x.state.size(); i++)
        df.state_max = std::max(df.state_max, (double)std::fabs(x.state[i] - y.state[i]));
    return df;
}

void report(const char* name, const Diff& d) {
    std::printf("%-58s identical=%s  max|dout|=%.3e  max|dstate|=%.3e\n", name, d.identical ? "yes" : "NO ",
                d.out_max, d.state_max);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("[SKIP] no GPU\n"); return 0; }
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);
    int failures = 0;

    {   // chunk scan, split at a chunk boundary
        Host h = make_inputs(512, 1);
        Dev d = to_device(h);
        const Run full = run(d, {0, 512}, false, st);
        const Diff carried = diff(full, run(d, {0, 256, 512}, true, st));
        const Diff dropped = diff(full, run(d, {0, 256, 512}, false, st));
        const Diff off_boundary = diff(full, run(d, {0, 200, 512}, true, st));
        report("chunk 512 = chunk 256 + chunk 256 (carry)", carried);
        report("control: same split, carry off", dropped);
        report("chunk 512 vs chunk 200 + chunk 312 (carry, off a boundary)", off_boundary);
        if (!carried.identical) { std::printf("FAIL: chunk carry does not reproduce the unsplit pass\n"); failures++; }
        // Judged on the outputs: a few hundred tokens on, the recurrence has forgotten its start, so the
        // final state barely moves -- the tokens right after the split are where a dropped state shows.
        if (dropped.out_max < 1e-3) { std::printf("FAIL: control did not diverge -- the test cannot see a dropped state\n"); failures++; }
        free_device(d);
    }
    {   // sequential scan (every pass under 128 tokens)
        Host h = make_inputs(96, 2);
        Dev d = to_device(h);
        const Run full = run(d, {0, 96}, false, st);
        const Diff carried = diff(full, run(d, {0, 40, 96}, true, st));
        const Diff dropped = diff(full, run(d, {0, 40, 96}, false, st));
        report("sequential 96 = sequential 40 + sequential 56 (carry)", carried);
        report("control: same split, carry off", dropped);
        if (!carried.identical) { std::printf("FAIL: sequential carry does not reproduce the unsplit pass\n"); failures++; }
        if (dropped.out_max < 1e-3) { std::printf("FAIL: control did not diverge\n"); failures++; }
        free_device(d);
    }
    {   // mixed: a resumed prefill's shape
        Host h = make_inputs(352, 3);
        Dev d = to_device(h);
        const Run full = run(d, {0, 352}, false, st);
        report("mixed: chunk 352 vs chunk 256 + sequential 96 (carry)", diff(full, run(d, {0, 256, 352}, true, st)));
        report("control: same split, carry off", diff(full, run(d, {0, 256, 352}, false, st)));
        free_device(d);
    }

    cudaStreamDestroy(st);
    std::printf(failures ? "gdn_prefill_carry_gpu_test: FAIL\n" : "gdn_prefill_carry_gpu_test: OK\n");
    return failures ? 1 : 0;
}
