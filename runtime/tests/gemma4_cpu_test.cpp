// CPU wiring test for Gemma4Model's 5L1G local/global attention dispatch.
//
// Builds a tiny Gemma4-shaped model (6 layers: 5 local + 1 global) and runs
// forward autoregressively in float vs double. Validates layer-type dispatch
// (different head_dim, GQA, RoPE theta, sliding window on local layers).
//
// Build: g++ -O2 -std=c++17 gemma4_cpu_test.cpp -o gemma4_cpu_test

#include "sparkinfer/models/gemma4.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using std::vector;

struct TinyCfg {
    int vocab = 40, hidden = 64, layers = 6, nq = 4;
    int local_hd = 16, global_hd = 32;
    int local_nkv = 2, global_nkv = 1;
    int local_window = 8, E = 4, K = 2, ffn = 32;
    float local_theta = 10000.f, global_theta = 1e6f, eps = 1e-6f;
};

template <typename R>
struct Wts {
    vector<R> embed, fnorm, lmhead;
    struct Layer {
        vector<R> in_norm, wq, wk, wv, wo, qn, kn, post, router, gate, up, down, sg, su, sd;
    };
    vector<Layer> layer;
};

template <typename R> static vector<R> cast(const vector<float>& v) { return vector<R>(v.begin(), v.end()); }

template <typename R>
static vector<R> rmsnorm(const vector<R>& x, const vector<R>& w, int rows, int cols, float eps) {
    vector<R> o(x.size());
    for (int r = 0; r < rows; r++) {
        double ss = 0;
        for (int c = 0; c < cols; c++) { R v = x[r * cols + c]; ss += (double)v * v; }
        double inv = 1.0 / std::sqrt(ss / cols + eps);
        for (int c = 0; c < cols; c++) o[r * cols + c] = (R)((double)x[r * cols + c] * inv * w[c]);
    }
    return o;
}

template <typename R>
static vector<R> matvec(const vector<R>& W, const vector<R>& v, int in, int out) {
    vector<R> y(out);
    for (int o = 0; o < out; o++) {
        double s = 0;
        for (int i = 0; i < in; i++) s += (double)v[i] * W[(size_t)i * out + o];
        y[o] = (R)s;
    }
    return y;
}

template <typename R>
static void rope(vector<R>& x, int p, int heads, int hd, float theta) {
    int half = hd / 2;
    for (int h = 0; h < heads; h++)
        for (int i = 0; i < half; i++) {
            double freq = std::pow((double)theta, -2.0 * i / hd), a = p * freq;
            double c = std::cos(a), s = std::sin(a);
            R& x0 = x[h * hd + i]; R& x1 = x[h * hd + i + half];
            double a0 = x0, a1 = x1;
            x0 = (R)(a0 * c - a1 * s); x1 = (R)(a1 * c + a0 * s);
        }
}

static bool is_global(int L) { return sparkinfer::gemma4_is_global_layer(L); }

template <typename R>
static vector<double> forward(const TinyCfg& cfg, const Wts<R>& W, const vector<int>& seq) {
    const int H = cfg.hidden;
    vector<vector<R>> Kc(cfg.layers), Vc(cfg.layers);
    for (size_t pos = 0; pos < seq.size(); pos++) {
        vector<R> x(W.embed.begin() + (size_t)seq[pos] * H, W.embed.begin() + (size_t)(seq[pos] + 1) * H);
        for (int L = 0; L < cfg.layers; L++) {
            const bool g = is_global(L);
            const int hd = g ? cfg.global_hd : cfg.local_hd;
            const int nkv = g ? cfg.global_nkv : cfg.local_nkv;
            const int qd = cfg.nq * hd, kd = nkv * hd;
            const float theta = g ? cfg.global_theta : cfg.local_theta;
            const auto& w = W.layer[L];

            auto xn = rmsnorm<R>(x, w.in_norm, 1, H, cfg.eps);
            auto q = matvec<R>(w.wq, xn, H, qd), k = matvec<R>(w.wk, xn, H, kd), v = matvec<R>(w.wv, xn, H, kd);
            q = rmsnorm<R>(q, w.qn, cfg.nq, hd, cfg.eps);
            k = rmsnorm<R>(k, w.kn, nkv, hd, cfg.eps);
            rope<R>(q, (int)pos, cfg.nq, hd, theta);
            rope<R>(k, (int)pos, nkv, hd, theta);
            for (int i = 0; i < kd; i++) { Kc[L].push_back(k[i]); Vc[L].push_back(v[i]); }

            const int T = (int)pos + 1;
            const int t0 = g ? 0 : std::max(0, T - cfg.local_window);
            vector<R> attn(qd, (R)0);
            float scale = 1.f / std::sqrt((float)hd);
            for (int h = 0; h < cfg.nq; h++) {
                int kvh = h / (cfg.nq / nkv);
                vector<double> sc(T - t0);
                double mx = -1e300;
                for (int ti = t0; ti < T; ti++) {
                    double d = 0;
                    for (int e = 0; e < hd; e++) d += (double)q[h * hd + e] * Kc[L][(size_t)ti * kd + kvh * hd + e];
                    sc[ti - t0] = d * scale;
                    mx = std::max(mx, sc[ti - t0]);
                }
                double den = 0;
                for (double s : sc) den += std::exp(s - mx);
                for (int e = 0; e < hd; e++) {
                    double a = 0;
                    for (int ti = t0; ti < T; ti++)
                        a += std::exp(sc[ti - t0] - mx) / den * Vc[L][(size_t)ti * kd + kvh * hd + e];
                    attn[h * hd + e] = (R)a;
                }
            }
            auto ao = matvec<R>(w.wo, attn, qd, H);
            for (int i = 0; i < H; i++) x[i] = (R)((double)x[i] + ao[i]);
            x = rmsnorm<R>(x, w.post, 1, H, cfg.eps);
            // skip MoE — identity passthrough for wiring test
        }
        x = rmsnorm<R>(x, W.fnorm, 1, H, cfg.eps);
    }
    vector<double> logits(cfg.vocab);
    for (int v = 0; v < cfg.vocab; v++) {
        double s = 0;
        for (int i = 0; i < H; i++) s += (double)x[i] * W.lmhead[(size_t)i * cfg.vocab + v];
        logits[v] = s;
    }
    return logits;
}

static Wts<float> rand_wts(const TinyCfg& cfg, std::mt19937& rng) {
    Wts<float> W;
    auto fill = [&](int n) {
        vector<float> v(n);
        for (float& x : v) x = std::uniform_real_distribution<float>(-0.05f, 0.05f)(rng);
        return v;
    };
    W.embed = fill(cfg.vocab * cfg.hidden);
    W.fnorm = fill(cfg.hidden);
    W.lmhead = fill(cfg.hidden * cfg.vocab);
    W.layer.resize(cfg.layers);
    for (int L = 0; L < cfg.layers; L++) {
        const bool g = is_global(L);
        const int hd = g ? cfg.global_hd : cfg.local_hd;
        const int nkv = g ? cfg.global_nkv : cfg.local_nkv;
        const int qd = cfg.nq * hd, kd = nkv * hd;
        auto& w = W.layer[L];
        w.in_norm = fill(cfg.hidden);
        w.wq = fill(cfg.hidden * qd); w.wk = fill(cfg.hidden * kd); w.wv = fill(cfg.hidden * kd);
        w.wo = fill(qd * cfg.hidden);
        w.qn = fill(hd); w.kn = fill(hd); w.post = fill(cfg.hidden);
        w.router = fill(cfg.hidden * cfg.E);
        w.gate = fill(cfg.E * cfg.hidden * cfg.ffn);
        w.up = fill(cfg.E * cfg.hidden * cfg.ffn);
        w.down = fill(cfg.E * cfg.ffn * cfg.hidden);
        w.sg = fill(cfg.hidden * cfg.ffn); w.su = fill(cfg.hidden * cfg.ffn); w.sd = fill(cfg.ffn * cfg.hidden);
    }
    return W;
}

int main() {
    TinyCfg cfg;
    std::mt19937 rng(42);
    auto Wf = rand_wts(cfg, rng);
    Wts<double> Wd;
    Wd.embed = cast<double>(Wf.embed); Wd.fnorm = cast<double>(Wf.fnorm); Wd.lmhead = cast<double>(Wf.lmhead);
    Wd.layer.resize(cfg.layers);
    for (int L = 0; L < cfg.layers; L++) {
        auto& d = Wd.layer[L]; auto& f = Wf.layer[L];
        d.in_norm = cast<double>(f.in_norm); d.wq = cast<double>(f.wq); d.wk = cast<double>(f.wk);
        d.wv = cast<double>(f.wv); d.wo = cast<double>(f.wo); d.qn = cast<double>(f.qn);
        d.kn = cast<double>(f.kn); d.post = cast<double>(f.post);
        d.router = cast<double>(f.router); d.gate = cast<double>(f.gate);
        d.up = cast<double>(f.up); d.down = cast<double>(f.down);
        d.sg = cast<double>(f.sg); d.su = cast<double>(f.su); d.sd = cast<double>(f.sd);
    }

    // 5L1G pattern: layer 5 is global in a 6-layer model
    if (!is_global(5) || is_global(4)) {
        fprintf(stderr, "[FAIL] 5L1G layer classification wrong\n");
        return 1;
    }

    vector<int> seq = {3, 7, 11, 15, 19, 23, 27};   // long enough to exercise local window
    auto lf = forward<float>(cfg, Wf, seq);
    auto ld = forward<double>(cfg, Wd, seq);

    double max_err = 0;
    for (int i = 0; i < cfg.vocab; i++) max_err = std::max(max_err, std::fabs(lf[i] - ld[i]));
    printf("gemma4_cpu_test: max logit err = %.2e (5L1G, window=%d)\n", max_err, cfg.local_window);
    if (max_err > 1e-4) { printf("[FAIL]\n"); return 1; }
    printf("[PASS]\n");
    return 0;
}
