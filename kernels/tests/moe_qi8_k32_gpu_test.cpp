// The routed BM=128 fused-decode GEMM on m16n8k32 with a pipelined weight stage
// (pfm_moe_gemm_qi8_k32_kernel) must decode the same int8 bytes and accumulate the same int32 dots
// as pfm_moe_gemm_qi8_kernel. A bf16-store launch (the gate / up form: gathered A rows, one C row
// per pair) is held bit for bit; a scatter launch (the down form: float atomicAdd per token) is held
// within float reordering, which the old kernel has run to run anyway.
//
// SPARKINFER_PREFILL_MOE_K32 is read once per process, so each arm runs in a forked child and pipes
// its outputs back. Experts carry 0 to 300 pairs, so a block may own a full 128-row tile, a partial
// one, or the second tile of an expert. Q4_K, Q5_K and Q6_K each run both forms.
#include "sparkinfer/kernels/prefill_moe_q.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

constexpr int E = 6, T = 400;                      // experts, tokens
const int kCnt[E] = {300, 0, 128, 37, 129, 6};     // pairs per expert
constexpr int BM = 128;

unsigned int rng = 0x13579bdu;
unsigned next_u() { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
float next_unit() { return ((int)(next_u() & 0xFFFF) - 32768) / 32768.f; }

// Random k-quant blocks with a sane fp16 d (and dmin): at the head for Q4_K (144 B) and Q5_K
// (176 B), at byte 208 for Q6_K (210 B, no dmin).
std::vector<unsigned char> kquant(size_t nblocks, size_t bs) {
    std::vector<unsigned char> w(nblocks * bs);
    for (auto& b : w) b = (unsigned char)next_u();
    for (size_t b = 0; b < nblocks; ++b) {
        const __half d = __float2half(0.002f + 0.002f * std::fabs(next_unit()));
        const __half m = __float2half(0.001f * std::fabs(next_unit()));
        if (bs == 210) {
            std::memcpy(&w[b * bs + 208], &d, 2);
        } else {
            std::memcpy(&w[b * bs], &d, 2);
            std::memcpy(&w[b * bs + 2], &m, 2);
        }
    }
    return w;
}

struct Case { int qtype; size_t bs; int n_out, K; bool scatter; };
const Case kCases[] = {
    {12, 144, 512, 2048, false},   // Q4_K gate / up shape
    {13, 176, 2048, 512, true},    // Q5_K down shape, scattered per token
    {12, 144, 2048, 512, true},    // Q4_K down
    {14, 210, 512, 2048, false},   // Q6_K, decoded from global (no register stage)
    {14, 210, 2048, 512, true},    // Q6_K down
};
constexpr int NC = sizeof(kCases) / sizeof(kCases[0]);

size_t out_floats(const Case& c, int P) { return c.scatter ? (size_t)T * c.n_out : (size_t)P * c.n_out; }

int child(int fd) {
    int P = 0;
    std::vector<int> offs(E + 1, 0);
    for (int e = 0; e < E; ++e) { offs[e] = P; P += kCnt[e]; }
    offs[E] = P;
    std::vector<int> tm;
    for (int e = 0; e < E; ++e)
        for (int mt = 0; mt * BM < kCnt[e]; ++mt) { tm.push_back(e); tm.push_back(mt); }
    const int ntiles = (int)tm.size() / 2, max_tiles = P / BM + E;
    std::vector<int> ptok(P);
    std::vector<float> pw(P);
    for (int p = 0; p < P; ++p) { ptok[p] = (int)(next_u() % T); pw[p] = 0.1f + 0.2f * std::fabs(next_unit()); }
    int *d_offs, *d_tm, *d_nt, *d_ptok;
    float *d_pw;
    bool ok = cudaMalloc(&d_offs, offs.size() * 4) == cudaSuccess && cudaMalloc(&d_tm, tm.size() * 4) == cudaSuccess &&
              cudaMalloc(&d_nt, 4) == cudaSuccess && cudaMalloc(&d_ptok, P * 4) == cudaSuccess &&
              cudaMalloc(&d_pw, P * 4) == cudaSuccess &&
              cudaMemcpy(d_offs, offs.data(), offs.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_tm, tm.data(), tm.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_nt, &ntiles, 4, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_ptok, ptok.data(), P * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_pw, pw.data(), P * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    for (int ci = 0; ok && ci < NC; ++ci) {
        const Case& c = kCases[ci];
        const int rows_a = c.scatter ? P : T;          // down: A is per pair (h); gate/up: per token
        std::vector<signed char> a((size_t)rows_a * c.K);
        for (auto& v : a) v = (signed char)((int)(next_u() % 255) - 127);
        std::vector<float> sx(rows_a), rs((size_t)E * c.n_out);
        for (auto& v : sx) v = 0.01f + 0.01f * std::fabs(next_unit());
        for (auto& v : rs) v = 0.02f + 0.02f * std::fabs(next_unit());
        auto w = kquant((size_t)E * c.n_out * (c.K / 256), c.bs);
        signed char* d_a; float *d_sx, *d_rs, *d_out = nullptr; unsigned char* d_w; __nv_bfloat16* d_c = nullptr;
        const size_t nout = out_floats(c, P);
        ok = cudaMalloc(&d_a, a.size()) == cudaSuccess && cudaMalloc(&d_sx, sx.size() * 4) == cudaSuccess &&
             cudaMalloc(&d_rs, rs.size() * 4) == cudaSuccess && cudaMalloc(&d_w, w.size()) == cudaSuccess &&
             cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(d_sx, sx.data(), sx.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(d_rs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok && c.scatter) ok = cudaMalloc(&d_out, nout * 4) == cudaSuccess && cudaMemset(d_out, 0, nout * 4) == cudaSuccess;
        if (ok && !c.scatter) ok = cudaMalloc(&d_c, nout * 2) == cudaSuccess && cudaMemset(d_c, 0xff, nout * 2) == cudaSuccess;
        ok = ok && sparkinfer::kernels::launch_pfm_moe_gemm_qi8(
                       c.qtype, d_a, d_sx, d_w, d_rs, d_ptok, d_pw, d_offs, d_tm, d_nt, d_c, d_out,
                       c.n_out, c.K, max_tiles, BM, /*a_indirect=*/!c.scatter, c.scatter, nullptr) &&
             cudaDeviceSynchronize() == cudaSuccess;
        std::vector<float> out(nout);
        if (ok && c.scatter) ok = cudaMemcpy(out.data(), d_out, nout * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        if (ok && !c.scatter) {
            std::vector<__nv_bfloat16> cb(nout);
            ok = cudaMemcpy(cb.data(), d_c, nout * 2, cudaMemcpyDeviceToHost) == cudaSuccess;
            for (size_t i = 0; i < nout; ++i) out[i] = __bfloat162float(cb[i]);
        }
        for (size_t wr = 0; ok && wr < nout * 4;) {
            const ssize_t n = write(fd, reinterpret_cast<const char*>(out.data()) + wr, nout * 4 - wr);
            if (n <= 0) ok = false; else wr += (size_t)n;
        }
        cudaFree(d_a); cudaFree(d_sx); cudaFree(d_rs); cudaFree(d_w); cudaFree(d_out); cudaFree(d_c);
    }
    return ok ? 0 : 1;
}

bool run(const char* k32, std::vector<std::vector<float>>& outs) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        setenv("SPARKINFER_PREFILL_MOE_K32", k32, 1);
        _exit(child(fds[1]));
    }
    close(fds[1]);
    int P = 0;
    for (int e = 0; e < E; ++e) P += kCnt[e];
    outs.assign(NC, {});
    bool ok = true;
    for (int ci = 0; ok && ci < NC; ++ci) {
        outs[ci].resize(out_floats(kCases[ci], P));
        size_t got = 0, want = outs[ci].size() * 4;
        for (ssize_t n; got < want && (n = read(fds[0], reinterpret_cast<char*>(outs[ci].data()) + got, want - got)) > 0;)
            got += (size_t)n;
        ok = got == want;
    }
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    return ok && WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

}  // namespace

int main() {
    {
        const pid_t pid = fork();
        if (pid == 0) { int n = 0; _exit(cudaGetDeviceCount(&n) == cudaSuccess && n > 0 ? 0 : 77); }
        int st = 0;
        waitpid(pid, &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { std::printf("[SKIP] no GPU\n"); return 77; }
    }
    std::vector<std::vector<float>> ref, got;
    if (!run("0", ref) || !run("1", got)) { std::printf("[FAIL] a run failed\n"); return 1; }
    bool ok = true;
    for (int ci = 0; ci < NC; ++ci) {
        const Case& c = kCases[ci];
        const auto& a = got[ci];
        const auto& b = ref[ci];
        size_t nonzero = 0;
        for (float v : b) nonzero += v != 0.f;
        if (nonzero < b.size() / 4) { std::printf("[FAIL] case %d: reference mostly zero\n", ci); ok = false; continue; }
        if (!c.scatter) {
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                std::printf("[FAIL] case %d (qtype %d, %dx%d): bf16 output differs\n", ci, c.qtype, c.n_out, c.K);
                ok = false;
            } else {
                std::printf("[PASS] case %d (qtype %d, %dx%d): bit-identical\n", ci, c.qtype, c.n_out, c.K);
            }
            continue;
        }
        double amax = 0, worst = 0;
        for (float v : b) amax = std::max(amax, (double)std::fabs(v));
        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs((double)a[i] - b[i]));
        if (worst > 1e-5 * amax) {
            std::printf("[FAIL] case %d (qtype %d scatter): max |diff| %.3g vs max %.3g\n", ci, c.qtype, worst, amax);
            ok = false;
        } else {
            std::printf("[PASS] case %d (qtype %d scatter): within float reordering (%.2g of max)\n", ci, c.qtype,
                        amax > 0 ? worst / amax : 0.0);
        }
    }
    if (!ok) return 1;
    std::printf("[PASS] moe_qi8_k32_gpu_test\n");
    return 0;
}
