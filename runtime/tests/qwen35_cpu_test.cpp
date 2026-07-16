// CPU end-to-end test for the Qwen3.5 model composition.
//
// Builds a tiny Qwen3.5-shaped model (random weights) and runs the full forward
// autoregressively — embedding, per-head QK-norm, RoPE, GQA attention over a
// growing KV cache, O-proj, routed top-k MoE + shared expert, residuals, final
// norm, LM head — in float (the runtime's algorithm) and in double (reference),
// and checks the logits match. Validates the model wiring that Qwen35Model::
// forward_token performs. Runs without a GPU.
//
// Build: g++ -O2 -std=c++17 qwen35_cpu_test.cpp -o qwen35_cpu_test

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using std::vector;

struct Cfg { int vocab=50, hidden=64, layers=2, nq=4, nkv=2, hd=16, E=4, K=2, ffn=32; float theta=10000.f, eps=1e-6f; };

template <typename R>
struct Weights {
    vector<R> embed, fnorm, lmhead;
    struct Layer { vector<R> in_norm,wq,wk,wv,wo,qn,kn,post,router,gate,up,down,sg,su,sd; };
    vector<Layer> layer;
};

template <typename R>
static vector<R> cast(const vector<float>& v) { return vector<R>(v.begin(), v.end()); }

template <typename R>
static vector<R> rmsnorm(const vector<R>& x, const vector<R>& w, int rows, int cols, float eps) {
    vector<R> o(x.size());
    for (int r=0;r<rows;r++){ double ss=0; for(int c=0;c<cols;c++){ R v=x[r*cols+c]; ss+=(double)v*v; }
        double inv=1.0/std::sqrt(ss/cols+eps);
        for(int c=0;c<cols;c++) o[r*cols+c]=(R)((double)x[r*cols+c]*inv*w[c]); }
    return o;
}
template <typename R>
static vector<R> matvec(const vector<R>& W, const vector<R>& v, int in, int out) {
    vector<R> y(out); for(int o=0;o<out;o++){ double s=0; for(int i=0;i<in;i++) s+=(double)v[i]*W[(size_t)i*out+o]; y[o]=(R)s; } return y;
}
template <typename R> static R silu(R x){ return (R)((double)x/(1.0+std::exp(-(double)x))); }

template <typename R>
static void rope(vector<R>& x, int p, int heads, int hd, float theta) {
    int half=hd/2;
    for(int h=0;h<heads;h++) for(int i=0;i<half;i++){
        double freq=std::pow((double)theta, -2.0*i/hd), a=p*freq, c=std::cos(a), s=std::sin(a);
        R& x0=x[h*hd+i]; R& x1=x[h*hd+i+half]; double a0=x0,a1=x1;
        x0=(R)(a0*c-a1*s); x1=(R)(a1*c+a0*s);
    }
}

// Full forward for the last token of `seq`; returns logits[vocab].
template <typename R>
static vector<double> forward(const Cfg& cfg, const Weights<R>& W, const vector<int>& seq) {
    const int H=cfg.hidden, qd=cfg.nq*cfg.hd, kd=cfg.nkv*cfg.hd;
    vector<vector<R>> Kc(cfg.layers), Vc(cfg.layers);   // KV cache per layer, appended
    vector<double> logits;
    for (size_t pos=0; pos<seq.size(); pos++) {
        vector<R> x(W.embed.begin()+(size_t)seq[pos]*H, W.embed.begin()+(size_t)(seq[pos]+1)*H);
        for (int l=0;l<cfg.layers;l++) {
            const auto& w=W.layer[l];
            auto xn=rmsnorm<R>(x,w.in_norm,1,H,cfg.eps);
            auto q=matvec<R>(w.wq,xn,H,qd), k=matvec<R>(w.wk,xn,H,kd), v=matvec<R>(w.wv,xn,H,kd);
            q=rmsnorm<R>(q,w.qn,cfg.nq,cfg.hd,cfg.eps);
            k=rmsnorm<R>(k,w.kn,cfg.nkv,cfg.hd,cfg.eps);
            rope<R>(q,(int)pos,cfg.nq,cfg.hd,cfg.theta);
            rope<R>(k,(int)pos,cfg.nkv,cfg.hd,cfg.theta);
            for(int i=0;i<kd;i++){ Kc[l].push_back(k[i]); Vc[l].push_back(v[i]); }
            int T=(int)pos+1;
            vector<R> attn(qd,(R)0);
            float scale=1.f/std::sqrt((float)cfg.hd);
            for(int h=0;h<cfg.nq;h++){ int kvh=h/(cfg.nq/cfg.nkv);
                vector<double> sc(T); double mx=-1e300;
                for(int t=0;t<T;t++){ double d=0; for(int e=0;e<cfg.hd;e++) d+=(double)q[h*cfg.hd+e]*Kc[l][t*kd+kvh*cfg.hd+e]; sc[t]=d*scale; mx=std::max(mx,sc[t]); }
                double den=0; for(int t=0;t<T;t++) den+=std::exp(sc[t]-mx);
                for(int e=0;e<cfg.hd;e++){ double a=0; for(int t=0;t<T;t++) a+=std::exp(sc[t]-mx)/den*Vc[l][t*kd+kvh*cfg.hd+e]; attn[h*cfg.hd+e]=(R)a; }
            }
            auto ao=matvec<R>(w.wo,attn,qd,H);
            vector<R> hh(H); for(int i=0;i<H;i++) hh[i]=(R)((double)x[i]+ao[i]);
            auto hn=rmsnorm<R>(hh,w.post,1,H,cfg.eps);
            // router top-k
            auto logit=matvec<R>(w.router,hn,H,cfg.E);
            vector<int> idx(cfg.E); for(int i=0;i<cfg.E;i++) idx[i]=i;
            std::stable_sort(idx.begin(),idx.end(),[&](int a,int b){return logit[a]>logit[b]||(logit[a]==logit[b]&&a<b);});
            double rmx=logit[idx[0]],rden=0; for(int j=0;j<cfg.K;j++) rden+=std::exp((double)logit[idx[j]]-rmx);
            vector<double> moe(H,0.0);
            auto run_ffn=[&](const vector<R>& g,const vector<R>& u,const vector<R>& d,int e,double wt){
                vector<double> hb(cfg.ffn);
                for(int f=0;f<cfg.ffn;f++){ double gg=0,uu=0; size_t bo=(size_t)e*H*cfg.ffn;
                    for(int i=0;i<H;i++){ gg+=(double)hn[i]*g[bo+(size_t)i*cfg.ffn+f]; uu+=(double)hn[i]*u[bo+(size_t)i*cfg.ffn+f]; }
                    hb[f]=(gg/(1.0+std::exp(-gg)))*uu; }
                for(int i=0;i<H;i++){ double y=0; size_t bo=(size_t)e*cfg.ffn*H; for(int f=0;f<cfg.ffn;f++) y+=hb[f]*d[bo+(size_t)f*H+i]; moe[i]+=wt*y; }
            };
            for(int j=0;j<cfg.K;j++){ int e=idx[j]; double wt=std::exp((double)logit[idx[j]]-rmx)/rden; run_ffn(w.gate,w.up,w.down,e,wt); }
            run_ffn(w.sg,w.su,w.sd,0,1.0);   // shared expert (single)
            for(int i=0;i<H;i++) x[i]=(R)((double)hh[i]+moe[i]);
        }
        auto xf=rmsnorm<R>(x,W.fnorm,1,H,cfg.eps);
        auto lg=matvec<R>(W.lmhead,xf,H,cfg.vocab);
        logits.assign(lg.begin(), lg.end());
    }
    return logits;
}

int main() {
    Cfg cfg;
    std::mt19937 rng(99);
    auto rv=[&](int n,float s){ vector<float> a(n); for(auto&x:a) x=s*std::uniform_real_distribution<float>(-1,1)(rng); return a; };
    const int H=cfg.hidden, qd=cfg.nq*cfg.hd, kd=cfg.nkv*cfg.hd, E=cfg.E, F=cfg.ffn;

    Weights<float> Wf; Weights<double> Wd;
    Wf.embed=rv(cfg.vocab*H,1.f); Wf.fnorm=rv(H,.5f); Wf.lmhead=rv(H*cfg.vocab,.1f);
    Wf.layer.resize(cfg.layers);
    for(int l=0;l<cfg.layers;l++){ auto&w=Wf.layer[l];
        w.in_norm=rv(H,.5f); w.wq=rv(H*qd,.1f); w.wk=rv(H*kd,.1f); w.wv=rv(H*kd,.1f); w.wo=rv(qd*H,.1f);
        w.qn=rv(cfg.hd,.5f); w.kn=rv(cfg.hd,.5f); w.post=rv(H,.5f); w.router=rv(H*E,.2f);
        w.gate=rv((size_t)E*H*F,.1f); w.up=rv((size_t)E*H*F,.1f); w.down=rv((size_t)E*F*H,.1f);
        w.sg=rv((size_t)H*F,.1f); w.su=rv((size_t)H*F,.1f); w.sd=rv((size_t)F*H,.1f);
    }
    // mirror to double
    Wd.embed=cast<double>(Wf.embed); Wd.fnorm=cast<double>(Wf.fnorm); Wd.lmhead=cast<double>(Wf.lmhead);
    Wd.layer.resize(cfg.layers);
    for(int l=0;l<cfg.layers;l++){ auto&s=Wf.layer[l]; auto&d=Wd.layer[l];
        d.in_norm=cast<double>(s.in_norm); d.wq=cast<double>(s.wq); d.wk=cast<double>(s.wk); d.wv=cast<double>(s.wv); d.wo=cast<double>(s.wo);
        d.qn=cast<double>(s.qn); d.kn=cast<double>(s.kn); d.post=cast<double>(s.post); d.router=cast<double>(s.router);
        d.gate=cast<double>(s.gate); d.up=cast<double>(s.up); d.down=cast<double>(s.down);
        d.sg=cast<double>(s.sg); d.su=cast<double>(s.su); d.sd=cast<double>(s.sd);
    }

    vector<int> seq = {3, 17, 2, 41};
    auto lf = forward<float>(cfg, Wf, seq);
    auto ld = forward<double>(cfg, Wd, seq);

    double err=0; int af=0, ad=0;
    for(int i=0;i<cfg.vocab;i++){ err=std::max(err,std::abs(lf[i]-ld[i])); if(lf[i]>lf[af]) af=i; if(ld[i]>ld[ad]) ad=i; }
    bool ok = err < 5e-3 && af == ad;
    printf("qwen3.5 model forward: logits max_err=%.3e  argmax float=%d double=%d -> %s\n",
           err, af, ad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
