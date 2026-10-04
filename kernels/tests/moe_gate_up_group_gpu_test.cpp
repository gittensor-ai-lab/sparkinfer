// The expert-grouped and the R-rows batched gate/up of Qwen3.6's routed MoE (hidden 2048, ffn 512,
// top-8, Q4_K gate/up, Q5_K down) must write h BIT-IDENTICAL to the one-warp-per-(pair, row) kernel
// they replace: they change which warp reads an expert's rows and how often, never what any
// (pair, row) computes. The grouped down sums a token's eight expert dots after each one's
// butterfly rather than before, so its output is held to the per-token kernel within bf16
// rounding, and to itself bit for bit across runs.
//
// Each variant is picked by environment variables that the launcher reads once per process, so
// every variant runs in its own forked child and reports, per case, a hash of h and the output;
// the parent compares them with the reference child's. The cases cover a spread routing, decode-like concentrated
// routing (~3 pairs an expert) and every token on the same eight experts (64-pair segments, many
// passes of the grouped kernel), at the batch widths a decode step hands the arm.
#include "sparkinfer/kernels/moe.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int H = 2048, F = 512, TOPK = 8, E = 64, MAXT = 64;
constexpr int ROUTINGS = 3;
constexpr int WIDTHS[] = {2, 5, 16, 32, 64};
constexpr int NCASES = ROUTINGS * (int)(sizeof(WIDTHS) / sizeof(WIDTHS[0]));

unsigned int rng = 0x1234567u;
unsigned char next_byte() {
    rng = rng * 1664525u + 1013904223u;
    return (unsigned char)(rng >> 24);
}
bool fill_dev(void* p, size_t bytes) {
    std::vector<unsigned char> h(bytes);
    for (size_t i = 0; i < bytes; ++i) h[i] = next_byte();
    return cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}
// Any byte pattern is a valid Q4_K block, but random half-precision scales can be inf/nan, which
// would make a bitwise compare vacuous; keep each block's d/dmin small and finite.
void tame_scales(std::vector<unsigned char>& w, size_t block) {
    for (size_t b = 0; b + block <= w.size(); b += block) {
        w[b + 0] = next_byte(); w[b + 1] = 0x1c;   // d    ~ 2^-8
        w[b + 2] = next_byte(); w[b + 3] = 0x18;   // dmin ~ 2^-9
    }
}
bool fill_kq(void* p, size_t bytes, size_t block) {   // Q4_K (144 B) or Q5_K (176 B): dm leads both
    std::vector<unsigned char> h(bytes);
    for (size_t i = 0; i < bytes; ++i) h[i] = next_byte();
    tame_scales(h, block);
    return cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

int route(int routing, int t, int k) {
    switch (routing) {
        case 0:  return (t * 37 + k * 11 + (t * k) % 5) % E;   // spread
        case 1:  return (t % 4) * 6 + (k * 3 + t / 4) % 24;    // concentrated, ~3 pairs an expert
        default: return k * 5;                                 // every token, the same 8 experts
    }
}

// Runs every case in this process; per case it writes an FNV-1a hash of h, then the bf16 output.
int child(int fd) {
    const size_t wbytes = (size_t)E * F * (H / 256) * 144;
    const size_t dbytes = (size_t)E * H * (F / 256) * 176;
    void *g = nullptr, *u = nullptr, *d = nullptr, *in = nullptr, *out = nullptr;
    float *hs = nullptr, *os = nullptr, *wts = nullptr;
    int* ids = nullptr;
    bool ok = cudaMalloc(&g, wbytes) == cudaSuccess && cudaMalloc(&u, wbytes) == cudaSuccess &&
              cudaMalloc(&d, dbytes) == cudaSuccess &&
              cudaMalloc(&in, (size_t)MAXT * H * sizeof(__nv_bfloat16)) == cudaSuccess &&
              cudaMalloc(&out, (size_t)MAXT * H * sizeof(__nv_bfloat16)) == cudaSuccess &&
              cudaMalloc(&hs, (size_t)MAXT * TOPK * F * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&os, (size_t)MAXT * TOPK * F * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&ids, MAXT * TOPK * sizeof(int)) == cudaSuccess &&
              cudaMalloc(&wts, MAXT * TOPK * sizeof(float)) == cudaSuccess;
    if (ok) {
        std::vector<__nv_bfloat16> x((size_t)MAXT * H);
        for (auto& v : x) v = __float2bfloat16(((int)next_byte() - 128) / 64.f);
        ok = fill_kq(g, wbytes, 144) && fill_kq(u, wbytes, 144) && fill_kq(d, dbytes, 176) &&
             cudaMemcpy(in, x.data(), x.size() * sizeof(x[0]), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    std::vector<float> h((size_t)MAXT * TOPK * F);
    std::vector<__nv_bfloat16> o((size_t)MAXT * H);
    for (int r = 0; ok && r < ROUTINGS; ++r) {
        for (int t : WIDTHS) {
            std::vector<int> hid(t * TOPK);
            std::vector<float> hw(t * TOPK, 1.f / TOPK);
            for (int i = 0; i < t; ++i)
                for (int k = 0; k < TOPK; ++k) hid[i * TOPK + k] = route(r, i, k);
            ok = cudaMemcpy(ids, hid.data(), hid.size() * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(wts, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemset(hs, 0xff, (size_t)t * TOPK * F * sizeof(float)) == cudaSuccess;
            if (!ok) break;
            sparkinfer::kernels::launch_moe_expert_ffn_q4k(in, g, u, d, 12, 12, 13, ids, wts, out, hs, os,
                                                           t, TOPK, H, F);
            ok = cudaDeviceSynchronize() == cudaSuccess &&
                 cudaMemcpy(h.data(), hs, (size_t)t * TOPK * F * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                 cudaMemcpy(o.data(), out, (size_t)t * H * sizeof(o[0]), cudaMemcpyDeviceToHost) == cudaSuccess;
            if (!ok) break;
            unsigned long long hash = 1469598103934665603ull;
            const unsigned char* p = reinterpret_cast<const unsigned char*>(h.data());
            for (size_t i = 0; i < (size_t)t * TOPK * F * sizeof(float); ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
            if (write(fd, &hash, sizeof(hash)) != (ssize_t)sizeof(hash)) ok = false;
            const size_t ob = (size_t)t * H * sizeof(o[0]);
            for (size_t w = 0; ok && w < ob;) {
                const ssize_t n = write(fd, reinterpret_cast<const unsigned char*>(o.data()) + w, ob - w);
                if (n <= 0) ok = false; else w += (size_t)n;
            }
        }
    }
    return ok ? 0 : 1;
}

struct Result {
    std::vector<unsigned long long> hash;          // per case
    std::vector<std::vector<float>> out;           // per case, the output as float
};

bool read_all(int fd, void* dst, size_t bytes) {
    size_t got = 0;
    for (ssize_t n; got < bytes && (n = read(fd, static_cast<unsigned char*>(dst) + got, bytes - got)) > 0;)
        got += (size_t)n;
    return got == bytes;
}

bool run_variant(const char* const* env, Result& res) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        setenv("SPARKINFER_MOE_GU_ROWS_MIN", "2", 1);   // every width here takes the batched arm
        for (int i = 0; env[i]; i += 2) setenv(env[i], env[i + 1], 1);
        _exit(child(fds[1]));
    }
    close(fds[1]);
    res.hash.assign(NCASES, 0);
    res.out.assign(NCASES, {});
    bool ok = true;
    for (int c = 0; ok && c < NCASES; ++c) {
        const int t = WIDTHS[c % (NCASES / ROUTINGS)];
        std::vector<__nv_bfloat16> o((size_t)t * H);
        ok = read_all(fds[0], &res.hash[c], sizeof(res.hash[c])) &&
             read_all(fds[0], o.data(), o.size() * sizeof(o[0]));
        res.out[c].resize(o.size());
        for (size_t i = 0; i < o.size(); ++i) res.out[c][i] = __bfloat162float(o[i]);
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace

int main() {
    int n = 0;
    {
        // probe in a child too: a CUDA context in the parent would not survive the forks
        const pid_t pid = fork();
        if (pid == 0) _exit(cudaGetDeviceCount(&n) == cudaSuccess && n > 0 ? 0 : 77);
        int st = 0;
        waitpid(pid, &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { std::printf("[SKIP] no GPU\n"); return 77; }
    }
    const char* ref_env[] = {"SPARKINFER_MOE_GU_ROWS", "1", "SPARKINFER_MOE_DOWN_GROUP", "0", nullptr};
    Result ref;
    if (!run_variant(ref_env, ref)) { std::printf("[FAIL] reference run\n"); return 1; }
    // down: "exact" = the per-token down ran (bitwise output check); otherwise the grouped one
    struct V { const char* label; bool exact_down; const char* env[9]; };
    const V variants[] = {
        {"grouped, 2 pairs a pass", false, {"SPARKINFER_MOE_GU_GROUP", "2", nullptr}},
        {"grouped, 4 pairs a pass", false, {"SPARKINFER_MOE_GU_GROUP", "4", nullptr}},
        {"grouped, 8 pairs a pass", false, {"SPARKINFER_MOE_GU_GROUP", "8", nullptr}},
        {"grouped gate/up, per-token down", true, {"SPARKINFER_MOE_DOWN_GROUP", "0", nullptr}},
        {"4 rows a warp, sorted",   false, {"SPARKINFER_MOE_GU_GROUP", "0", "SPARKINFER_MOE_GU_ROWS", "4", nullptr}},
        {"2 rows a warp, unsorted", true, {"SPARKINFER_MOE_GU_GROUP", "0", "SPARKINFER_MOE_GU_SORT", "0",
                                           "SPARKINFER_MOE_GU_ROWS", "2", nullptr}},
    };
    bool ok = true;
    Result grouped_first;
    for (const V& v : variants) {
        Result got;
        if (!run_variant(v.env, got)) { std::printf("[FAIL] %s: run failed\n", v.label); ok = false; continue; }
        int bad = 0;
        double worst = 0.0;
        for (int c = 0; c < NCASES; ++c) {
            const int width = WIDTHS[c % (NCASES / ROUTINGS)], routing = c / (NCASES / ROUTINGS);
            if (got.hash[c] != ref.hash[c]) {
                std::printf("[FAIL] %s: routing %d width %d: h differs from the per-pair kernel\n", v.label,
                            routing, width);
                ++bad;
            }
            const std::vector<float>& a = got.out[c];
            const std::vector<float>& b = ref.out[c];
            if (v.exact_down) {
                if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                    std::printf("[FAIL] %s: routing %d width %d: output differs\n", v.label, routing, width);
                    ++bad;
                }
                continue;
            }
            // grouped down: the same terms summed in another order, rounded to bf16
            double scale = 0.0;
            for (float x : b) scale = std::max(scale, (double)std::fabs(x));
            for (size_t i = 0; i < a.size(); ++i) {
                const double err = std::fabs((double)a[i] - (double)b[i]);
                const double tol = 1.6e-2 * std::fabs((double)b[i]) + 1e-4 * scale;
                worst = std::max(worst, err / (scale > 0 ? scale : 1.0));
                if (!(err <= tol)) {
                    std::printf("[FAIL] %s: routing %d width %d: out[%zu] %g vs %g\n", v.label, routing, width,
                                i, a[i], b[i]);
                    ++bad;
                    break;
                }
            }
        }
        if (!v.exact_down) {
            // and the grouped down is deterministic: every grouped variant writes the same output
            if (grouped_first.out.empty()) grouped_first = got;
            else
                for (int c = 0; c < NCASES; ++c)
                    if (std::memcmp(got.out[c].data(), grouped_first.out[c].data(),
                                    got.out[c].size() * sizeof(float)) != 0) {
                        std::printf("[FAIL] %s: case %d: grouped down output not reproducible\n", v.label, c);
                        ++bad;
                    }
        }
        if (!bad)
            std::printf("[PASS] %s: h bit-identical, output %s in all %d cases (worst |err|/max %.2e)\n", v.label,
                        v.exact_down ? "bit-identical" : "within bf16 rounding", NCASES, worst);
        ok = ok && !bad;
    }
    if (!ok) return 1;
    std::printf("[PASS] moe_gate_up_group_gpu_test\n");
    return 0;
}
