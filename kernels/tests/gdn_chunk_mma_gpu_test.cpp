// The register-resident Gated-DeltaNet prefill scan (pf_gdnc_scan_mma_kernel) against the block
// form (pf_gdnc_scan_kernel) and against an fp64 CPU run of the recurrence itself:
//     S_t = g_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,   y_t = S_t^T q_t / sqrt(HD)
// Neither GPU form is exact (both narrow S and U~ to bf16 for the tensor cores), so each is held to
// the fp64 reference with the same tolerance, and the two are held to each other at 8K, where the
// reference would take minutes. Two, four and eight warps a block must give the same bits. Shapes: Qwen3.8-27B (16 q/k heads, 48 v-heads) and
// Qwen3.6-35B-A3B (16 / 32), a short prompt with a partial final chunk, and a second call that
// carries the state in.
//
// SPARKINFER_PREFILL_GDN_SCAN_MMA is read once per process, so each arm runs in a forked child and
// pipes its outputs back. The 8K timing per arm is printed for reference.
#include "sparkinfer/kernels/prefill_gdn_chunk.h"
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

constexpr int HD = 128;

unsigned int rng = 0x2468aceu;
unsigned next_u() { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
float next_unit() { return ((int)(next_u() & 0xFFFF) - 32768) / 32768.f; }

struct Shape { int qh, vh, n1, n2; bool ref; };   // n2 > 0: a second call carrying the state
const Shape kShapes[] = {
    {16, 48, 200, 0, true},       // Qwen3.8, partial final chunk, fp64 reference
    {16, 32, 300, 150, true},     // Qwen3.6, carry-in, both calls with tails
    {16, 48, 8192, 0, false},     // Qwen3.8 8K: arms against each other, timed
    {16, 32, 8192, 0, false},     // Qwen3.6 8K
};

struct Inputs {
    std::vector<__nv_bfloat16> q, k, v, al, be, dt, a;
};

Inputs make_inputs(int N, int qh, int vh) {
    Inputs in;
    in.q.resize((size_t)N * qh * HD); in.k.resize(in.q.size()); in.v.resize((size_t)N * vh * HD);
    // q and k are L2-normalized per head before the scan in the model; do the same here.
    for (int t = 0; t < N; t++)
        for (int hh = 0; hh < qh; hh++) {
            float bq[HD], bk[HD], nq = 0.f, nk = 0.f;
            for (int d = 0; d < HD; d++) {
                bq[d] = next_unit(); bk[d] = next_unit();
                nq += bq[d] * bq[d]; nk += bk[d] * bk[d];
            }
            nq = 1.f / std::sqrt(nq); nk = 1.f / std::sqrt(nk);
            for (int d = 0; d < HD; d++) {
                in.q[((size_t)t * qh + hh) * HD + d] = __float2bfloat16(bq[d] * nq);
                in.k[((size_t)t * qh + hh) * HD + d] = __float2bfloat16(bk[d] * nk);
            }
        }
    for (auto& x : in.v) x = __float2bfloat16(next_unit());
    in.al.resize((size_t)N * vh); in.be.resize(in.al.size());
    for (auto& x : in.al) x = __float2bfloat16(3.f * next_unit());
    for (auto& x : in.be) x = __float2bfloat16(3.f * next_unit());
    in.dt.resize(vh); in.a.resize(vh);
    for (int hh = 0; hh < vh; hh++) {
        in.dt[hh] = __float2bfloat16(next_unit());
        in.a[hh] = __float2bfloat16(-std::exp(2.5f * next_unit() - 1.5f));   // A = -exp(A_log) < 0
    }
    return in;
}

template <class T>
T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

// One arm: run every shape, write out + final state (as float) to fd; time the 8K shapes.
int run_arm(int fd, const char* mode) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    setenv("SPARKINFER_PREFILL_GDN_SCAN_MMA", mode, 1);
    rng = 0x2468aceu;
    for (const Shape& sh : kShapes) {
        const int ncall = sh.n2 > 0 ? 2 : 1;
        float* state = nullptr;
        cudaMalloc(&state, (size_t)sh.vh * HD * HD * sizeof(float));
        cudaMemset(state, 0, (size_t)sh.vh * HD * HD * sizeof(float));
        for (int call = 0; call < ncall; call++) {
            const int N = call ? sh.n2 : sh.n1;
            Inputs in = make_inputs(N, sh.qh, sh.vh);
            auto *q = dev_copy(in.q), *k = dev_copy(in.k), *v = dev_copy(in.v);
            auto *al = dev_copy(in.al), *be = dev_copy(in.be), *dt = dev_copy(in.dt), *a = dev_copy(in.a);
            __nv_bfloat16* out = nullptr;
            cudaMalloc(&out, (size_t)N * sh.vh * HD * sizeof(__nv_bfloat16));
            if (!sparkinfer::kernels::launch_prefill_gdn_chunk(q, k, v, al, be, dt, a, state, out, N,
                                                              sh.qh, sh.vh, HD, true, nullptr,
                                                              call > 0, 0)) {
                fprintf(stderr, "launch refused (mode %s, N=%d)\n", mode, N);
                return 1;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            if (!sh.ref && N >= 4096) {
                std::vector<float> st0((size_t)sh.vh * HD * HD);
                cudaMemcpy(st0.data(), state, st0.size() * 4, cudaMemcpyDeviceToHost);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0); cudaEventCreate(&e1);
                constexpr int R = 20;
                cudaEventRecord(e0);
                for (int r = 0; r < R; r++)
                    sparkinfer::kernels::launch_prefill_gdn_chunk(q, k, v, al, be, dt, a, state, out,
                                                                  N, sh.qh, sh.vh, HD, true,
                                                                  nullptr, false, 0);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                printf("  mode %-2s q/v heads %d/%d N=%d: %.1f us a call (prep + scan)\n", mode, sh.qh,
                       sh.vh, N, 1000.f * ms / R);
                cudaMemcpy(state, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice);
            }
            std::vector<__nv_bfloat16> ho((size_t)N * sh.vh * HD);
            cudaMemcpy(ho.data(), out, ho.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            std::vector<float> fo(ho.size());
            for (size_t i = 0; i < ho.size(); i++) fo[i] = __bfloat162float(ho[i]);
            if (write(fd, fo.data(), fo.size() * 4) != (ssize_t)(fo.size() * 4)) return 1;
            cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(al); cudaFree(be); cudaFree(dt);
            cudaFree(a); cudaFree(out);
        }
        std::vector<float> hs((size_t)sh.vh * HD * HD);
        cudaMemcpy(hs.data(), state, hs.size() * 4, cudaMemcpyDeviceToHost);
        if (write(fd, hs.data(), hs.size() * 4) != (ssize_t)(hs.size() * 4)) return 1;
        cudaFree(state);
    }
    return 0;
}

bool read_all(int fd, std::vector<float>& dst) {
    std::vector<char> buf;
    char tmp[1 << 16];
    ssize_t r;
    while ((r = read(fd, tmp, sizeof tmp)) > 0) buf.insert(buf.end(), tmp, tmp + r);
    dst.resize(buf.size() / 4);
    std::memcpy(dst.data(), buf.data(), dst.size() * 4);
    return true;
}

// 0 ran, 77 no device, else failed. The parent never touches CUDA: a forked child cannot use a
// driver its parent initialized.
int run_child(const char* mode, std::vector<float>& res) {
    int p[2];
    if (pipe(p)) return 1;
    const pid_t pid = fork();
    if (pid == 0) {
        close(p[0]);
        const int rc = run_arm(p[1], mode);
        fflush(stdout);
        _exit(rc);
    }
    close(p[1]);
    read_all(p[0], res);
    close(p[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

// fp64 sequential reference for one shape: outputs of every call, then the final state, in the
// same order run_arm writes them.
std::vector<float> reference(const Shape& sh) {
    std::vector<float> res;
    std::vector<double> S((size_t)sh.vh * HD * HD, 0.0);    // [h][row m][col j]
    const int ncall = sh.n2 > 0 ? 2 : 1;
    for (int call = 0; call < ncall; call++) {
        const int N = call ? sh.n2 : sh.n1;
        Inputs in = make_inputs(N, sh.qh, sh.vh);
        std::vector<float> out((size_t)N * sh.vh * HD);
        for (int h = 0; h < sh.vh; h++) {
            const int qh = h / (sh.vh / sh.qh);                       // qh_block = true
            const double a_h = __bfloat162float(in.a[h]), dt_h = __bfloat162float(in.dt[h]);
            double* Sh = &S[(size_t)h * HD * HD];
            for (int t = 0; t < N; t++) {
                const double x = __bfloat162float(in.al[(size_t)t * sh.vh + h]) + dt_h;
                const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
                const double g = std::exp(sp * a_h);
                const double b = 1.0 / (1.0 + std::exp(-(double)__bfloat162float(in.be[(size_t)t * sh.vh + h])));
                double kk[HD], qq[HD], vv[HD], kS[HD];
                for (int d = 0; d < HD; d++) {
                    kk[d] = __bfloat162float(in.k[((size_t)t * sh.qh + qh) * HD + d]);
                    qq[d] = __bfloat162float(in.q[((size_t)t * sh.qh + qh) * HD + d]);
                    vv[d] = __bfloat162float(in.v[((size_t)t * sh.vh + h) * HD + d]);
                }
                for (int j = 0; j < HD; j++) {
                    double acc = 0.0;
                    for (int m = 0; m < HD; m++) acc += kk[m] * Sh[m * HD + j];
                    kS[j] = acc;
                }
                for (int m = 0; m < HD; m++)
                    for (int j = 0; j < HD; j++)
                        Sh[m * HD + j] = g * (Sh[m * HD + j] - b * kk[m] * kS[j]) + b * kk[m] * vv[j];
                for (int j = 0; j < HD; j++) {
                    double acc = 0.0;
                    for (int m = 0; m < HD; m++) acc += qq[m] * Sh[m * HD + j];
                    out[((size_t)t * sh.vh + h) * HD + j] = (float)(acc / std::sqrt((double)HD));
                }
            }
        }
        res.insert(res.end(), out.begin(), out.end());
    }
    for (int h = 0; h < sh.vh; h++)                                   // [h][col][row] like the GPU
        for (int j = 0; j < HD; j++)
            for (int m = 0; m < HD; m++) res.push_back((float)S[((size_t)h * HD + m) * HD + j]);
    return res;
}

// Relative error of x against r: max |x - r| / max |r|, and the rms of x - r over the rms of r.
void rel_err(const float* x, const float* r, size_t n, double& mx, double& rms) {
    double dmax = 0, rmax = 0, d2 = 0, r2 = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)x[i] - r[i];
        dmax = std::max(dmax, std::fabs(d));
        rmax = std::max(rmax, std::fabs((double)r[i]));
        d2 += d * d;
        r2 += (double)r[i] * r[i];
    }
    mx = rmax > 0 ? dmax / rmax : dmax;
    rms = r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2);
}

}  // namespace

int main() {
    std::vector<float> old_r, new_r;
    const int rc = run_child("0", old_r);
    if (rc == 77) {
        printf("SKIP: no CUDA device\n");
        return 77;
    }
    if (rc != 0 || run_child("4", new_r) != 0 || old_r.size() != new_r.size() || old_r.empty()) {
        printf("FAIL: an arm did not run (%zu / %zu floats)\n", old_r.size(), new_r.size());
        return 1;
    }
    for (const char* extra : {"2", "8"}) {
        std::vector<float> x;
        if (run_child(extra, x) != 0 || x.size() != new_r.size()) {
            printf("FAIL: arm %s did not run\n", extra);
            return 1;
        }
        double mx, rms;
        rel_err(x.data(), new_r.data(), x.size(), mx, rms);
        printf("  warps %s vs 4: max %.2e rms %.2e\n", extra, mx, rms);
        if (mx != 0.0) { printf("FAIL: block width changed the values\n"); return 1; }
    }

    bool ok = true;
    rng = 0x2468aceu;
    size_t off = 0;
    for (const Shape& sh : kShapes) {
        const int ncall = sh.n2 > 0 ? 2 : 1;
        size_t n_out = (size_t)sh.n1 * sh.vh * HD + (ncall > 1 ? (size_t)sh.n2 * sh.vh * HD : 0);
        const size_t n_st = (size_t)sh.vh * HD * HD;
        double mo, ro, ms, rs;
        rel_err(new_r.data() + off, old_r.data() + off, n_out, mo, ro);
        rel_err(new_r.data() + off + n_out, old_r.data() + off + n_out, n_st, ms, rs);
        printf("q/v %d/%d N=%d%s: new vs old  out max %.2e rms %.2e, state max %.2e rms %.2e\n",
               sh.qh, sh.vh, sh.n1, ncall > 1 ? " +carry" : "", mo, ro, ms, rs);
        if (ro > 2e-2 || rs > 2e-2) ok = false;
        if (sh.ref) {
            const std::vector<float> ref = reference(sh);
            if (ref.size() != n_out + n_st) { printf("FAIL: reference size\n"); return 1; }
            for (int arm = 0; arm < 2; arm++) {
                const float* x = (arm ? new_r : old_r).data() + off;
                rel_err(x, ref.data(), n_out, mo, ro);
                rel_err(x + n_out, ref.data() + n_out, n_st, ms, rs);
                printf("  %s vs fp64: out max %.2e rms %.2e, state max %.2e rms %.2e\n",
                       arm ? "new" : "old", mo, ro, ms, rs);
                if (ro > 2e-2 || rs > 2e-2 || mo > 1e-1) ok = false;
            }
        } else {
            // keep the generator in step with run_arm
            for (int call = 0; call < ncall; call++) make_inputs(call ? sh.n2 : sh.n1, sh.qh, sh.vh);
        }
        off += n_out + n_st;
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
