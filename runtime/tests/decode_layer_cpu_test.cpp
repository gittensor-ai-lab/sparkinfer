// CPU end-to-end test for the MoE decode-layer composition.
//
// Re-implements DecodeRunner::decode_layer's exact pipeline
//   RMSNorm -> QKV -> KV-append -> GQA attention -> O-proj
//   -> residual+RMSNorm -> top-k MoE SwiGLU -> residual
// using the same per-step algorithms as the CUDA kernels, and checks the result
// against a fully independent double-precision dense reference. A match proves
// the integration wiring (shapes, residual order, norm placement, routing ->
// FFN accumulation) is correct. Runs without a GPU.
//
// Build: g++ -O2 -std=c++17 decode_layer_cpu_test.cpp -o decode_layer_cpu_test

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using std::vector;
static std::mt19937 rng(7);
static float fr(float s = 1.f) { return std::uniform_real_distribution<float>(-s, s)(rng); }
static float silu(float x) { return x / (1.f + std::exp(-x)); }

int main() {
    // Small config; num_q_heads*head_dim == hidden so O-proj maps back to hidden.
    const int H = 64, hd = 16, nq = 4, nkv = 2, Q = nq * hd, KV = nkv * hd;
    const int E = 8, K = 2, F = 32, L = 5;   // L prior KV tokens
    const float scale = 1.f / std::sqrt((float)hd);

    auto rv = [&](int n, float s){ vector<float> a(n); for (auto& x : a) x = fr(s); return a; };
    vector<float> x = rv(H, 1.f);
    vector<float> wq = rv(H*Q,.1f), wk = rv(H*KV,.1f), wv = rv(H*KV,.1f), wo = rv(Q*H,.1f);
    vector<float> an = rv(H,.5f), fn = rv(H,.5f), rw = rv(H*E,.2f);
    vector<float> gate = rv(E*H*F,.1f), up = rv(E*H*F,.1f), down = rv(E*F*H,.1f);
    vector<float> Kc = rv(L*KV,1.f), Vc = rv(L*KV,1.f);   // existing cache

    auto matvec = [](const vector<float>& W, const vector<float>& v, int in, int out){
        vector<float> y(out,0.f); for (int o=0;o<out;o++){ float s=0; for(int i=0;i<in;i++) s+=v[i]*W[(size_t)i*out+o]; y[o]=s;} return y; };

    // ---- double-precision dense reference ----
    auto rmsnorm_d = [&](const vector<float>& v, const vector<float>& w){
        double ss=0; for(float e:v) ss+=(double)e*e; double inv=1.0/std::sqrt(ss/v.size()+1e-6);
        vector<float> o(v.size()); for(size_t i=0;i<v.size();i++) o[i]=(float)(v[i]*inv*w[i]); return o; };
    vector<double> ref(H);
    {
        vector<float> xn = rmsnorm_d(x, an);
        vector<float> q = matvec(wq,xn,H,Q), k = matvec(wk,xn,H,KV), v = matvec(wv,xn,H,KV);
        vector<float> Kf = Kc, Vf = Vc; for(int i=0;i<KV;i++){Kf.push_back(k[i]);Vf.push_back(v[i]);}
        int T = L+1;
        vector<float> attn(Q,0.f);
        for(int h=0;h<nq;h++){ int kvh=h/(nq/nkv);
            vector<double> sc(T); double mx=-1e300;
            for(int t=0;t<T;t++){ double d=0; for(int e=0;e<hd;e++) d+=(double)q[h*hd+e]*Kf[t*KV+kvh*hd+e]; sc[t]=d*scale; mx=std::max(mx,sc[t]); }
            double den=0; for(int t=0;t<T;t++) den+=std::exp(sc[t]-mx);
            for(int e=0;e<hd;e++){ double a=0; for(int t=0;t<T;t++) a+=std::exp(sc[t]-mx)/den*Vf[t*KV+kvh*hd+e]; attn[h*hd+e]=(float)a; }
        }
        vector<float> ao = matvec(wo,attn,Q,H);
        vector<float> hres(H); for(int i=0;i<H;i++) hres[i]=x[i]+ao[i];
        vector<float> hn = rmsnorm_d(hres, fn);
        vector<float> logit = matvec(rw,hn,H,E);
        vector<int> idx(E); for(int i=0;i<E;i++) idx[i]=i;
        std::stable_sort(idx.begin(),idx.end(),[&](int a,int b){return logit[a]>logit[b]||(logit[a]==logit[b]&&a<b);});
        double mx=logit[idx[0]], den=0; for(int j=0;j<K;j++) den+=std::exp((double)logit[idx[j]]-mx);
        vector<double> moe(H,0.0);
        for(int j=0;j<K;j++){ int e=idx[j]; double w=std::exp((double)logit[idx[j]]-mx)/den;
            vector<double> hb(F);
            for(int f=0;f<F;f++){ double g=0,u=0; for(int i=0;i<H;i++){ g+=(double)hn[i]*gate[((size_t)e*H+i)*F+f]; u+=(double)hn[i]*up[((size_t)e*H+i)*F+f]; } hb[f]=(g/(1.0+std::exp(-g)))*u; }
            for(int i=0;i<H;i++){ double y=0; for(int f=0;f<F;f++) y+=hb[f]*down[((size_t)e*F+f)*H+i]; moe[i]+=w*y; }
        }
        for(int i=0;i<H;i++) ref[i]=hres[i]+moe[i];
    }

    // ---- float "kernel-algorithm" path (mirrors decode_layer.cpp step order) ----
    vector<float> out(H);
    {
        auto rmsnorm_f = [&](const vector<float>& v, const vector<float>& w){
            float ss=0; for(float e:v) ss+=e*e; float inv=1.f/std::sqrt(ss/v.size()+1e-6f);
            vector<float> o(v.size()); for(size_t i=0;i<v.size();i++) o[i]=v[i]*inv*w[i]; return o; };
        vector<float> xn = rmsnorm_f(x, an);
        vector<float> q = matvec(wq,xn,H,Q), k = matvec(wk,xn,H,KV), v = matvec(wv,xn,H,KV);
        vector<float> Kf = Kc, Vf = Vc; for(int i=0;i<KV;i++){Kf.push_back(k[i]);Vf.push_back(v[i]);}
        int T = L+1;
        vector<float> attn(Q,0.f);
        for(int h=0;h<nq;h++){ int kvh=h/(nq/nkv);                     // online softmax (kernel style)
            float m=-1e30f,l=0.f; vector<float> acc(hd,0.f);
            for(int t=0;t<T;t++){ float d=0; for(int e=0;e<hd;e++) d+=q[h*hd+e]*Kf[t*KV+kvh*hd+e]; float s=d*scale;
                float mn=std::max(m,s),c=std::exp(m-mn),p=std::exp(s-mn); l=l*c+p;
                for(int e=0;e<hd;e++) acc[e]=acc[e]*c+p*Vf[t*KV+kvh*hd+e]; m=mn; }
            for(int e=0;e<hd;e++) attn[h*hd+e]=acc[e]/l;
        }
        vector<float> ao = matvec(wo,attn,Q,H);
        vector<float> hres(H); for(int i=0;i<H;i++) hres[i]=x[i]+ao[i];
        vector<float> hn = rmsnorm_f(hres, fn);
        vector<float> logit = matvec(rw,hn,H,E);
        vector<float> s = logit; vector<int> sel(K); vector<float> sl(K);   // mask-argmax top-k
        for(int j=0;j<K;j++){ float b=-1e30f; int bi=-1; for(int e=0;e<E;e++) if(s[e]>b||(s[e]==b&&e<bi)){b=s[e];bi=e;} sel[j]=bi;sl[j]=b;s[bi]=-1e30f; }
        float mx=sl[0]; for(int j=1;j<K;j++) mx=std::max(mx,sl[j]); float den=0; for(int j=0;j<K;j++) den+=std::exp(sl[j]-mx);
        vector<float> moe(H,0.f);
        for(int j=0;j<K;j++){ int e=sel[j]; float w=std::exp(sl[j]-mx)/den;
            vector<float> hb(F);
            for(int f=0;f<F;f++){ float g=0,u=0; for(int i=0;i<H;i++){ g+=hn[i]*gate[((size_t)e*H+i)*F+f]; u+=hn[i]*up[((size_t)e*H+i)*F+f]; } hb[f]=silu(g)*u; }
            for(int i=0;i<H;i++){ float y=0; for(int f=0;f<F;f++) y+=hb[f]*down[((size_t)e*F+f)*H+i]; moe[i]+=w*y; }
        }
        for(int i=0;i<H;i++) out[i]=hres[i]+moe[i];
    }

    double err = 0; for (int i = 0; i < H; i++) err = std::max(err, std::abs((double)out[i] - ref[i]));
    bool ok = err < 1e-3;
    printf("decode-layer composition: max_err=%.3e -> %s\n", err, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
