// DFlash draft runtime: safetensors load + block-parallel forward.
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/models/dflash_kernels.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparkinfer {
namespace {

using bf16 = __nv_bfloat16;

inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        fprintf(stderr, "[dflash] %s: %s\n", what, cudaGetErrorString(e));
}

struct TensorView {
    void* data = nullptr;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
};

// Minimal safetensors reader (BF16 / F32 only). Owns a host mmap-like buffer.
struct SafeTensorsFile {
    std::vector<char> bytes;
    std::unordered_map<std::string, TensorView> tensors; // pointers into bytes

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        const std::streamoff sz = f.tellg();
        if (sz < 8) return false;
        f.seekg(0, std::ios::beg);
        bytes.resize((size_t)sz);
        f.read(bytes.data(), sz);
        if (!f) return false;
        uint64_t hdr_len = 0;
        memcpy(&hdr_len, bytes.data(), 8);
        if (8 + hdr_len > (uint64_t)sz) return false;
        const std::string hdr(bytes.data() + 8, bytes.data() + 8 + hdr_len);
        // Parse each "name":{...} entry for dtype/shape/data_offsets.
        size_t pos = 0;
        while (pos < hdr.size()) {
            size_t key_start = hdr.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = hdr.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            std::string key = hdr.substr(key_start + 1, key_end - key_start - 1);
            // Tensor entries are always "name": { ... }
            size_t colon = hdr.find(':', key_end);
            if (colon == std::string::npos) break;
            size_t obj = hdr.find('{', colon);
            if (obj == std::string::npos || obj > colon + 4) {
                // Not a tensor object (e.g. string metadata field) — advance past this key.
                pos = key_end + 1;
                continue;
            }
            if (key == "__metadata__") {
                // Skip nested object.
                int depth = 0;
                size_t i = obj;
                for (; i < hdr.size(); i++) {
                    if (hdr[i] == '{') depth++;
                    else if (hdr[i] == '}') {
                        depth--;
                        if (depth == 0) { i++; break; }
                    }
                }
                pos = i;
                continue;
            }
            size_t obj_end = hdr.find('}', obj);
            if (obj_end == std::string::npos) break;
            std::string body = hdr.substr(obj, obj_end - obj + 1);
            // dtype
            std::string dtype;
            size_t d0 = body.find("\"dtype\"");
            if (d0 != std::string::npos) {
                size_t c = body.find(':', d0);
                size_t q0 = body.find('"', c);
                size_t q1 = body.find('"', q0 + 1);
                if (q0 != std::string::npos && q1 != std::string::npos)
                    dtype = body.substr(q0 + 1, q1 - q0 - 1);
            }
            // data_offsets: [start, end]
            int64_t off0 = 0, off1 = 0;
            size_t o0 = body.find("\"data_offsets\"");
            if (o0 != std::string::npos) {
                size_t b0 = body.find('[', o0);
                off0 = strtoll(body.c_str() + b0 + 1, nullptr, 10);
                size_t comma = body.find(',', b0);
                off1 = strtoll(body.c_str() + comma + 1, nullptr, 10);
            }
            std::vector<int64_t> shape;
            size_t s0 = body.find("\"shape\"");
            if (s0 != std::string::npos) {
                size_t b0 = body.find('[', s0);
                size_t b1 = body.find(']', b0);
                std::string ss = body.substr(b0 + 1, b1 - b0 - 1);
                size_t p = 0;
                while (p < ss.size()) {
                    while (p < ss.size() && (ss[p] == ' ' || ss[p] == ',')) p++;
                    if (p >= ss.size()) break;
                    shape.push_back(strtoll(ss.c_str() + p, nullptr, 10));
                    while (p < ss.size() && ss[p] != ',') p++;
                }
            }
            if (dtype != "BF16" && dtype != "F32" && dtype != "BOOL") {
                fprintf(stderr, "[dflash] skip tensor %s dtype=%s\n", key.c_str(), dtype.c_str());
                pos = obj_end + 1;
                continue;
            }
            TensorView tv;
            tv.shape = shape;
            tv.nbytes = (size_t)(off1 - off0);
            tv.data = bytes.data() + 8 + hdr_len + off0;
            tensors[key] = tv;
            pos = obj_end + 1;
        }
        if (tensors.empty())
            fprintf(stderr, "[dflash] safetensors parse produced 0 tensors (hdr_len=%llu)\n",
                    (unsigned long long)hdr_len);
        return !tensors.empty();
    }
};

bool parse_config_json(const std::string& path, DFlashDraftConfig& cfg) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string j = ss.str();
    auto find_int = [&](const char* key, int& dst) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = j.find(pat);
        if (p == std::string::npos) return;
        size_t c = j.find(':', p);
        dst = (int)strtol(j.c_str() + c + 1, nullptr, 10);
    };
    auto find_float = [&](const char* key, float& dst) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = j.find(pat);
        if (p == std::string::npos) return;
        size_t c = j.find(':', p);
        dst = strtof(j.c_str() + c + 1, nullptr);
    };
    find_int("hidden_size", cfg.hidden);
    find_int("intermediate_size", cfg.intermediate);
    find_int("num_hidden_layers", cfg.n_layers);
    find_int("num_attention_heads", cfg.n_q_heads);
    find_int("num_key_value_heads", cfg.n_kv_heads);
    find_int("head_dim", cfg.head_dim);
    find_int("vocab_size", cfg.vocab);
    find_int("sliding_window", cfg.sliding_window);
    find_float("rms_norm_eps", cfg.rms_eps);
    // rope_theta nested
    size_t rp = j.find("\"rope_theta\"");
    if (rp != std::string::npos) {
        size_t c = j.find(':', rp);
        cfg.rope_theta = strtof(j.c_str() + c + 1, nullptr);
    }
    // dflash_config.block_size / mask_token_id / target_layer_ids
    size_t df = j.find("\"dflash_config\"");
    if (df != std::string::npos) {
        size_t bs = j.find("\"block_size\"", df);
        if (bs != std::string::npos) {
            size_t c = j.find(':', bs);
            cfg.block_size = (int)strtol(j.c_str() + c + 1, nullptr, 10);
        }
        size_t mt = j.find("\"mask_token_id\"", df);
        if (mt != std::string::npos) {
            size_t c = j.find(':', mt);
            cfg.mask_token_id = (int)strtol(j.c_str() + c + 1, nullptr, 10);
        }
        size_t tl = j.find("\"target_layer_ids\"", df);
        if (tl != std::string::npos) {
            size_t b0 = j.find('[', tl);
            size_t b1 = j.find(']', b0);
            cfg.target_layer_ids.clear();
            size_t p = b0 + 1;
            while (p < b1) {
                while (p < b1 && (j[p] == ' ' || j[p] == ',')) p++;
                if (p >= b1) break;
                cfg.target_layer_ids.push_back((int)strtol(j.c_str() + p, nullptr, 10));
                while (p < b1 && j[p] != ',') p++;
            }
        }
    }
    // layer_types
    size_t lt = j.find("\"layer_types\"");
    if (lt != std::string::npos) {
        size_t b0 = j.find('[', lt);
        size_t b1 = j.find(']', b0);
        cfg.sliding_layers.assign(cfg.n_layers, true);
        int idx = 0;
        size_t p = b0;
        while (p < b1 && idx < cfg.n_layers) {
            size_t q0 = j.find('"', p);
            if (q0 == std::string::npos || q0 >= b1) break;
            size_t q1 = j.find('"', q0 + 1);
            std::string t = j.substr(q0 + 1, q1 - q0 - 1);
            cfg.sliding_layers[idx++] = (t == "sliding_attention");
            p = q1 + 1;
        }
    } else {
        cfg.sliding_layers.assign(cfg.n_layers, true);
        if (cfg.n_layers > 0) cfg.sliding_layers[cfg.n_layers - 1] = false;
    }
    return true;
}

struct LayerWeights {
    bf16 *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    bf16 *q_norm = nullptr, *k_norm = nullptr;
    bf16 *input_norm = nullptr, *post_norm = nullptr;
    bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
};

} // namespace

struct DFlashDraftModel::Impl {
    DFlashDraftConfig cfg;
    std::vector<LayerWeights> layers;
    bf16* fc = nullptr;           // [H, n_cap * H] as [out, in] for gemv
    bf16* hidden_norm = nullptr;
    bf16* final_norm = nullptr;
    std::vector<void*> owned;

    // Shared target pointers
    const void* embed = nullptr;
    const void* lm_head = nullptr;
    int lm_head_type = 0;
    int vocab = 0;
    int hidden = 0;

    // Scratch
    cudaStream_t stream{};
    bf16 *noise = nullptr;          // [B, H]
    bf16 *target_proj = nullptr;    // [ctx, H]
    bf16 *x = nullptr, *xn = nullptr, *h = nullptr, *hn = nullptr;
    bf16 *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr, *ao = nullptr;
    bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
    bf16 *k_new = nullptr, *v_new = nullptr;  // [ctx+B, n_kv, d]
    float* logits = nullptr;        // [B, vocab]
    int *d_ids = nullptr, *d_out = nullptr;
    int *h_out = nullptr;

    // Per-layer contiguous KV cache: [max_seq, n_kv, d]
    std::vector<bf16*> k_cache, v_cache;
    int seq_len = 0;

    template <class T> T* alloc(size_t n) {
        void* p = nullptr;
        cu(cudaMalloc(&p, n * sizeof(T)), "malloc");
        owned.push_back(p);
        return (T*)p;
    }

    bf16* upload(const TensorView& tv) {
        bf16* d = alloc<bf16>(tv.nbytes / sizeof(bf16));
        if (tv.nbytes % sizeof(bf16) == 0) {
            cu(cudaMemcpy(d, tv.data, tv.nbytes, cudaMemcpyHostToDevice), "upload bf16");
        } else {
            // F32 -> BF16
            size_t n = tv.nbytes / sizeof(float);
            std::vector<bf16> tmp(n);
            const float* src = (const float*)tv.data;
            for (size_t i = 0; i < n; i++) tmp[i] = __float2bfloat16(src[i]);
            cu(cudaMemcpy(d, tmp.data(), n * sizeof(bf16), cudaMemcpyHostToDevice), "upload f32");
        }
        return d;
    }
};

DFlashDraftModel::DFlashDraftModel(const DFlashDraftConfig& cfg) : p_(new Impl()) {
    p_->cfg = cfg;
    if (p_->cfg.sliding_layers.empty()) {
        p_->cfg.sliding_layers.assign(p_->cfg.n_layers, true);
        if (p_->cfg.n_layers > 0) p_->cfg.sliding_layers.back() = false;
    }
    cudaStreamCreate(&p_->stream);
}

DFlashDraftModel::~DFlashDraftModel() {
    if (!p_) return;
    for (void* p : p_->owned) cudaFree(p);
    if (p_->h_out) cudaFreeHost(p_->h_out);
    if (p_->stream) cudaStreamDestroy(p_->stream);
    delete p_;
    p_ = nullptr;
}

const DFlashDraftConfig& DFlashDraftModel::config() const { return p_->cfg; }

void DFlashDraftModel::set_shared_weights(const void* embed, const void* lm_head,
                                         int lm_head_type, int vocab, int hidden) {
    p_->embed = embed;
    p_->lm_head = lm_head;
    p_->lm_head_type = lm_head_type;
    p_->vocab = vocab;
    p_->hidden = hidden;
}

void DFlashDraftModel::reset() { p_->seq_len = 0; }

void DFlashDraftModel::crop(int keep) {
    if (keep < 0) keep = 0;
    if (keep > p_->seq_len) keep = p_->seq_len;
    p_->seq_len = keep;
}

int DFlashDraftModel::seq_len() const { return p_->seq_len; }

const float* DFlashDraftModel::last_logits() const { return p_->logits; }

bool DFlashDraftModel::load(const std::string& dir) {
    Impl& s = *p_;
    const std::string cfg_path = dir + "/config.json";
    const std::string st_path = dir + "/model.safetensors";
    parse_config_json(cfg_path, s.cfg);
    SafeTensorsFile st;
    if (!st.load(st_path)) {
        fprintf(stderr, "[dflash] failed to load %s\n", st_path.c_str());
        return false;
    }
    auto require = [&](const std::string& name) -> TensorView* {
        auto it = st.tensors.find(name);
        if (it == st.tensors.end()) {
            fprintf(stderr, "[dflash] missing tensor %s\n", name.c_str());
            return nullptr;
        }
        return &it->second;
    };

    const int H = s.cfg.hidden;
    const int I = s.cfg.intermediate;
    const int n_cap = (int)s.cfg.target_layer_ids.size();
    const int B = s.cfg.block_size;
    const int max_seq = s.cfg.max_seq;
    const int qdim = s.cfg.n_q_heads * s.cfg.head_dim;
    const int kvdim = s.cfg.n_kv_heads * s.cfg.head_dim;

    auto* fc = require("fc.weight");
    auto* hn = require("hidden_norm.weight");
    auto* nn = require("norm.weight");
    if (!fc || !hn || !nn) return false;
    s.fc = s.upload(*fc);
    s.hidden_norm = s.upload(*hn);
    s.final_norm = s.upload(*nn);

    s.layers.resize(s.cfg.n_layers);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        auto& lw = s.layers[L];
        const std::string pfx = "layers." + std::to_string(L) + ".";
        auto* wq = require(pfx + "self_attn.q_proj.weight");
        auto* wk = require(pfx + "self_attn.k_proj.weight");
        auto* wv = require(pfx + "self_attn.v_proj.weight");
        auto* wo = require(pfx + "self_attn.o_proj.weight");
        auto* qn = require(pfx + "self_attn.q_norm.weight");
        auto* kn = require(pfx + "self_attn.k_norm.weight");
        auto* in = require(pfx + "input_layernorm.weight");
        auto* pn = require(pfx + "post_attention_layernorm.weight");
        auto* g = require(pfx + "mlp.gate_proj.weight");
        auto* u = require(pfx + "mlp.up_proj.weight");
        auto* d = require(pfx + "mlp.down_proj.weight");
        if (!wq || !wk || !wv || !wo || !qn || !kn || !in || !pn || !g || !u || !d)
            return false;
        lw.wq = s.upload(*wq); lw.wk = s.upload(*wk); lw.wv = s.upload(*wv); lw.wo = s.upload(*wo);
        lw.q_norm = s.upload(*qn); lw.k_norm = s.upload(*kn);
        lw.input_norm = s.upload(*in); lw.post_norm = s.upload(*pn);
        lw.gate = s.upload(*g); lw.up = s.upload(*u); lw.down = s.upload(*d);
    }

    // Scratch
    const int max_ctx = max_seq;
    const int max_kv = max_ctx + B;
    s.noise = s.alloc<bf16>((size_t)B * H);
    s.target_proj = s.alloc<bf16>((size_t)max_ctx * H);
    s.x = s.alloc<bf16>((size_t)B * H);
    s.xn = s.alloc<bf16>((size_t)B * H);
    s.h = s.alloc<bf16>((size_t)B * H);
    s.hn = s.alloc<bf16>((size_t)B * H);
    s.q = s.alloc<bf16>((size_t)B * qdim);
    s.attn = s.alloc<bf16>((size_t)B * qdim);
    s.ao = s.alloc<bf16>((size_t)B * H);
    s.gate = s.alloc<bf16>((size_t)B * I);
    s.up = s.alloc<bf16>((size_t)B * I);
    s.down = s.alloc<bf16>((size_t)B * H);
    s.k_new = s.alloc<bf16>((size_t)max_kv * kvdim);
    s.v_new = s.alloc<bf16>((size_t)max_kv * kvdim);
    s.logits = s.alloc<float>((size_t)B * std::max(s.cfg.vocab, 1));
    s.d_ids = s.alloc<int>(B);
    s.d_out = s.alloc<int>(B);
    cu(cudaHostAlloc(&s.h_out, B * sizeof(int), cudaHostAllocDefault), "h_out");

    s.k_cache.resize(s.cfg.n_layers);
    s.v_cache.resize(s.cfg.n_layers);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        s.k_cache[L] = s.alloc<bf16>((size_t)max_seq * kvdim);
        s.v_cache[L] = s.alloc<bf16>((size_t)max_seq * kvdim);
    }

    s.seq_len = 0;
    fprintf(stderr, "[dflash] loaded draft: layers=%d H=%d B=%d n_cap=%d mask=%d\n",
            s.cfg.n_layers, H, B, n_cap, s.cfg.mask_token_id);
    (void)I; (void)n_cap;
    return true;
}

bool DFlashDraftModel::forward_block(const void* target_hidden, int ctx_len,
                                     const int* noise_ids, int pos0,
                                     int* out_argmax, cudaStream_t stream) {
    Impl& s = *p_;
    if (!s.fc || !s.embed || !s.lm_head || !noise_ids || !out_argmax) return false;
    if (ctx_len < 0 || ctx_len + s.cfg.block_size > s.cfg.max_seq + s.cfg.block_size) return false;
    cudaStream_t st = stream ? stream : s.stream;
    const auto& c = s.cfg;
    const int H = c.hidden;
    const int I = c.intermediate;
    const int B = c.block_size;
    const int n_cap = (int)c.target_layer_ids.size();
    const int qdim = c.n_q_heads * c.head_dim;
    const int kvdim = c.n_kv_heads * c.head_dim;
    const int d = c.head_dim;
    const float scale = 1.f / sqrtf((float)d);
    const int past = s.seq_len;

    cu(cudaMemcpyAsync(s.d_ids, noise_ids, B * sizeof(int), cudaMemcpyHostToDevice, st), "ids");
    kernels::launch_embedding(s.d_ids, s.embed, s.noise, B, H, st);

    // target_hidden [ctx, n_cap*H] -> fc -> hidden_norm -> target_proj [ctx, H]
    // fc.weight is [H, n_cap*H] (out, in). Loop gemv per row.
    if (ctx_len > 0) {
        const bf16* th = (const bf16*)target_hidden;
        for (int t = 0; t < ctx_len; t++) {
            kernels::launch_gemv(th + (size_t)t * n_cap * H, s.fc,
                                 s.target_proj + (size_t)t * H, H, n_cap * H, st);
        }
        dflash_kernels::launch_rms(s.target_proj, s.hidden_norm, s.target_proj,
                                   ctx_len, H, c.rms_eps, st);
    }

    // x = noise embedding
    cu(cudaMemcpyAsync(s.x, s.noise, (size_t)B * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
       "noise->x");

    for (int L = 0; L < c.n_layers; L++) {
        const LayerWeights& w = s.layers[L];
        dflash_kernels::launch_rms(s.x, w.input_norm, s.xn, B, H, c.rms_eps, st);

        // Q from noise, K/V from cat(target, noise)
        for (int t = 0; t < B; t++) {
            kernels::launch_gemv(s.xn + (size_t)t * H, w.wq, s.q + (size_t)t * qdim, qdim, H, st);
        }
        // Build k_new / v_new = cat(k_ctx, k_noise)
        for (int t = 0; t < ctx_len; t++) {
            kernels::launch_gemv(s.target_proj + (size_t)t * H, w.wk,
                                 s.k_new + (size_t)t * kvdim, kvdim, H, st);
            kernels::launch_gemv(s.target_proj + (size_t)t * H, w.wv,
                                 s.v_new + (size_t)t * kvdim, kvdim, H, st);
        }
        for (int t = 0; t < B; t++) {
            kernels::launch_gemv(s.xn + (size_t)t * H, w.wk,
                                 s.k_new + (size_t)(ctx_len + t) * kvdim, kvdim, H, st);
            kernels::launch_gemv(s.xn + (size_t)t * H, w.wv,
                                 s.v_new + (size_t)(ctx_len + t) * kvdim, kvdim, H, st);
        }

        const int new_len = ctx_len + B;
        // Q / K RMSNorm per head
        dflash_kernels::launch_rms_heads(s.q, w.q_norm, B, c.n_q_heads, d, c.rms_eps, st);
        dflash_kernels::launch_rms_heads(s.k_new, w.k_norm, new_len, c.n_kv_heads, d, c.rms_eps, st);

        // RoPE: Q at positions past..(past+B) if past≈pos0 for noise-only positions.
        // Match reference: position_ids cover past_len .. start+block_size for the cat length.
        // k positions: past .. past+new_len-1 when past==seq_len and we're appending.
        // Absolute: k_pos0 = pos0 - ctx_len (context features align with tokens just before noise),
        // q_pos0 = pos0.
        const int k_pos0 = pos0 - ctx_len;
        const int q_pos0 = pos0;
        dflash_kernels::launch_rope_seq(s.q, B, c.n_q_heads, d, q_pos0, c.rope_theta, st);
        dflash_kernels::launch_rope_seq(s.k_new, new_len, c.n_kv_heads, d, k_pos0, c.rope_theta, st);

        // Append into cache then attend over full past+new
        // Cache currently has `past` tokens. New keys start at offset `past`.
        if (past + new_len > c.max_seq) {
            fprintf(stderr, "[dflash] KV overflow past=%d new=%d max=%d\n", past, new_len, c.max_seq);
            return false;
        }
        cu(cudaMemcpyAsync(s.k_cache[L] + (size_t)past * kvdim, s.k_new,
                           (size_t)new_len * kvdim * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "k append");
        cu(cudaMemcpyAsync(s.v_cache[L] + (size_t)past * kvdim, s.v_new,
                           (size_t)new_len * kvdim * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "v append");
        const int kv_len = past + new_len;
        const int window = (L < (int)c.sliding_layers.size() && c.sliding_layers[L])
                               ? c.sliding_window : 0;
        dflash_kernels::launch_attn_gqa(s.q, s.k_cache[L], s.v_cache[L], s.attn,
                                        B, kv_len, c.n_q_heads, c.n_kv_heads, d,
                                        q_pos0, /*k_pos0_cache=*/0, window, scale, st);

        for (int t = 0; t < B; t++)
            kernels::launch_gemv(s.attn + (size_t)t * qdim, w.wo, s.ao + (size_t)t * H, H, qdim, st);
        dflash_kernels::launch_add(s.x, s.ao, s.h, B * H, st);

        dflash_kernels::launch_rms(s.h, w.post_norm, s.hn, B, H, c.rms_eps, st);
        for (int t = 0; t < B; t++) {
            kernels::launch_gemv(s.hn + (size_t)t * H, w.gate, s.gate + (size_t)t * I, I, H, st);
            kernels::launch_gemv(s.hn + (size_t)t * H, w.up,   s.up   + (size_t)t * I, I, H, st);
        }
        dflash_kernels::launch_swiglu(s.gate, s.up, s.gate, B * I, st);
        for (int t = 0; t < B; t++)
            kernels::launch_gemv(s.gate + (size_t)t * I, w.down, s.down + (size_t)t * H, H, I, st);
        dflash_kernels::launch_add(s.h, s.down, s.x, B * H, st);
    }

    dflash_kernels::launch_rms(s.x, s.final_norm, s.xn, B, H, c.rms_eps, st);

    // LM head (target weights) -> logits / argmax. Skip row 0 for proposals but still compute.
    const int V = s.vocab > 0 ? s.vocab : c.vocab;
    for (int t = 0; t < B; t++) {
        const bf16* row = s.xn + (size_t)t * H;
        float* logit_row = s.logits + (size_t)t * V;
        if (s.lm_head_type)
            kernels::launch_gemv_q_f32(row, s.lm_head, s.lm_head_type, logit_row, V, H, st);
        else
            kernels::launch_gemv_f32(row, s.lm_head, logit_row, V, H, st);
    }
    kernels::launch_argmax(s.logits, s.d_out, B, V, st);
    cu(cudaMemcpyAsync(s.h_out, s.d_out, B * sizeof(int), cudaMemcpyDeviceToHost, st), "argmax");
    cu(cudaStreamSynchronize(st), "draft sync");
    for (int t = 0; t < B; t++) out_argmax[t] = s.h_out[t];

    // Crop draft KV to pos0 (= start of block) — discards noise, keeps context features
    // when past was already pos0 and ctx filled the gap. Matches reference crop(start).
    crop(pos0);
    return true;
}

} // namespace sparkinfer
