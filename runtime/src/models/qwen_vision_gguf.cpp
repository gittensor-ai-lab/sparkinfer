// The Qwen vision tower loaded from a llama.cpp-style `mmproj` GGUF instead of the safetensors
// checkpoint (issue #1093).
//
// The tower itself is unchanged: an mmproj carries the SAME encoder the checkpoint does, under
// llama.cpp's clip names and with the dims written [in, out] rather than HF's [out, in]. Both
// loaders fill one QwenVisionWeights, so qwen_vision_forward and the preprocessing never learn
// which file the weights came from.
//
// Two differences from the safetensors path are real rather than cosmetic:
//
//   * an mmproj may be quantized (the one this was developed against is Q8_0 with F32 biases and
//     F16 in places), where model.visual.* is always bf16. Every tensor is dequantized to bf16 on
//     the device here, so everything downstream sees exactly what it saw before.
//   * the Conv3d patch embedding arrives SPLIT along its temporal axis, as `v.patch_embd.weight`
//     and `v.patch_embd.weight.1`. HF stores one [H, C, T, P, P] tensor, whose row is C-major with
//     the two time steps adjacent, so rebuilding it is an interleave per channel -- not a
//     concatenation of the two halves, which would put every t=1 patch after every t=0 one and
//     quietly convolve the wrong pixels.
#include "sparkinfer/models/qwen_vision.h"

#include "sparkinfer/gguf.h"
#include "sparkinfer/kernels/quant.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace sparkinfer {

namespace {

using bf16 = unsigned short;

float bf16_to_f32(bf16 h) { unsigned u = (unsigned)h << 16; float f; std::memcpy(&f, &u, 4); return f; }

bool cu_ok(cudaError_t e, const std::string& what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = what + ": " + cudaGetErrorString(e);
    return false;
}

// Dequantize one GGUF tensor to bf16 on the device. `want` is checked rather than trusted: a
// silently wrong shape here becomes a tower that runs and returns plausible nonsense.
const void* up(const GGUF& g, const std::string& name, long want,
               QwenVisionWeights& w, std::string& err) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) { err = "missing vision tensor " + name; return nullptr; }
    if (want > 0 && t->n_values != want) {
        err = name + ": expected " + std::to_string(want) + " values, got " +
              std::to_string(t->n_values);
        return nullptr;
    }
    if (!kernels::ggml_dequant_supported(t->ggml_type)) {
        err = name + ": unsupported ggml type " + std::to_string(t->ggml_type);
        return nullptr;
    }
    void* raw = nullptr;
    if (!cu_ok(cudaMalloc(&raw, (size_t)t->n_bytes), "cudaMalloc raw " + name, err)) return nullptr;
    if (!cu_ok(cudaMemcpy(raw, t->data, (size_t)t->n_bytes, cudaMemcpyHostToDevice),
               "upload " + name, err)) { cudaFree(raw); return nullptr; }
    void* dst = nullptr;
    if (!cu_ok(cudaMalloc(&dst, (size_t)t->n_values * sizeof(bf16)), "cudaMalloc " + name, err)) {
        cudaFree(raw); return nullptr;
    }
    kernels::launch_gguf_dequant(t->ggml_type, raw, static_cast<bf16*>(dst), t->n_values, nullptr);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaFree(raw);
    if (!cu_ok(e, "dequant " + name, err)) { cudaFree(dst); return nullptr; }
    w.owned.push_back(dst);
    return dst;
}

}  // namespace

bool qwen_vision_config_from_gguf(const GGUF& g, QwenVisionConfig& cfg) {
    if (g.meta_str("general.architecture") != "clip") return false;
    if (g.meta_int("clip.has_vision_encoder", 0) != 1) return false;
    // Defaults stay whatever the struct ships: a key an mmproj omits is one this model does not
    // vary, and zeroing it would be worse than the released value.
    const auto set = [&](const char* key, int& dst) {
        const long v = g.meta_int(key, 0);
        if (v > 0) dst = (int)v;
    };
    set("clip.vision.block_count", cfg.depth);
    set("clip.vision.embedding_length", cfg.hidden);
    set("clip.vision.attention.head_count", cfg.n_heads);
    set("clip.vision.feed_forward_length", cfg.intermediate);
    set("clip.vision.patch_size", cfg.patch_size);
    set("clip.vision.spatial_merge_size", cfg.spatial_merge);
    set("clip.vision.projection_dim", cfg.out_hidden);
    // The learned position table's size is not a metadata key -- it is the tensor's own first
    // dimension, and the tower needs it to be a perfect square to resample on a patch grid.
    if (const GGUFTensor* pos = g.tensor("v.position_embd.weight")) {
        if (cfg.hidden > 0 && pos->n_values % cfg.hidden == 0)
            cfg.num_pos_embeddings = (int)(pos->n_values / cfg.hidden);
    }
    cfg.present = true;
    return true;
}

bool load_qwen_vision_weights_gguf(const GGUF& g, const QwenVisionConfig& cfg,
                                   QwenVisionWeights& w, std::string& err) {
    const long H = cfg.hidden, I = cfg.intermediate;
    const long M = cfg.merged_patch_dim();
    const long per_t = (long)cfg.in_channels * cfg.patch_size * cfg.patch_size;   // one time slice
    const long patch_in = per_t * cfg.temporal_patch;

    // ---- patch embedding: two time slices, interleaved per channel (see the header note) ----
    {
        const GGUFTensor* t0 = g.tensor("v.patch_embd.weight");
        const GGUFTensor* t1 = g.tensor("v.patch_embd.weight.1");
        if (!t0) { err = "missing vision tensor v.patch_embd.weight"; return false; }
        if (t1 && cfg.temporal_patch != 2) {
            err = "v.patch_embd.weight.1 present but temporal_patch != 2";
            return false;
        }
        if (t0->n_values != H * per_t) {
            err = "v.patch_embd.weight: expected " + std::to_string(H * per_t) + " values, got " +
                  std::to_string(t0->n_values);
            return false;
        }
        if (t1 && t1->n_values != t0->n_values) {
            err = "v.patch_embd.weight.1 disagrees in size with v.patch_embd.weight";
            return false;
        }
        // Dequantize both halves to host bf16, then weave them into HF's [H][C][T][P][P] order.
        std::vector<bf16> h0, h1, joined;
        auto to_host = [&](const GGUFTensor* t, std::vector<bf16>& out) -> bool {
            QwenVisionWeights tmp;
            const void* d = nullptr;
            {
                // Reuse `up` for the dequant, then pull it back and release the device copy: the
                // interleave is a host-side shuffle and this runs once at load.
                d = up(g, t == t0 ? "v.patch_embd.weight" : "v.patch_embd.weight.1",
                       t->n_values, tmp, err);
                if (!d) return false;
            }
            out.resize((size_t)t->n_values);
            const cudaError_t e = cudaMemcpy(out.data(), d, out.size() * sizeof(bf16),
                                             cudaMemcpyDeviceToHost);
            for (void* p : tmp.owned) cudaFree(p);
            return cu_ok(e, "readback patch_embd", err);
        };
        if (!to_host(t0, h0)) return false;
        if (t1 && !to_host(t1, h1)) return false;

        const long PP = (long)cfg.patch_size * cfg.patch_size;
        joined.assign((size_t)H * patch_in, 0);
        for (long h = 0; h < H; ++h)
            for (long c = 0; c < cfg.in_channels; ++c)
                for (int tt = 0; tt < cfg.temporal_patch; ++tt) {
                    const std::vector<bf16>& src = (tt == 0 || !t1) ? h0 : h1;
                    const bf16* s = src.data() + (size_t)h * per_t + (size_t)c * PP;
                    bf16* d = joined.data() + (size_t)h * patch_in +
                              ((size_t)c * cfg.temporal_patch + tt) * PP;
                    std::memcpy(d, s, (size_t)PP * sizeof(bf16));
                }
        void* dev = nullptr;
        if (!cu_ok(cudaMalloc(&dev, joined.size() * sizeof(bf16)), "cudaMalloc patch_w", err))
            return false;
        if (!cu_ok(cudaMemcpy(dev, joined.data(), joined.size() * sizeof(bf16),
                              cudaMemcpyHostToDevice), "upload patch_w", err)) {
            cudaFree(dev); return false;
        }
        w.owned.push_back(dev);
        w.patch_w = dev;
    }
    w.patch_b = up(g, "v.patch_embd.bias", H, w, err); if (!w.patch_b) return false;

    // ---- position table, host-side for the same reason as the safetensors path ----
    {
        const GGUFTensor* pos = g.tensor("v.position_embd.weight");
        if (!pos || pos->n_values != (long)cfg.num_pos_embeddings * H) {
            err = "v.position_embd.weight missing or wrong size"; return false;
        }
        w.pos_side = (int)llround(std::sqrt((double)cfg.num_pos_embeddings));
        if ((long)w.pos_side * w.pos_side != cfg.num_pos_embeddings) {
            err = "position table is not a perfect square"; return false;
        }
        QwenVisionWeights tmp;
        const void* d = up(g, "v.position_embd.weight", pos->n_values, tmp, err);
        if (!d) return false;
        std::vector<bf16> host((size_t)pos->n_values);
        const cudaError_t e = cudaMemcpy(host.data(), d, host.size() * sizeof(bf16),
                                         cudaMemcpyDeviceToHost);
        for (void* p : tmp.owned) cudaFree(p);
        if (!cu_ok(e, "readback position_embd", err)) return false;
        w.pos_table.resize(host.size());
        for (size_t i = 0; i < host.size(); ++i) w.pos_table[i] = bf16_to_f32(host[i]);
    }

    w.blocks.resize(cfg.depth);
    for (int b = 0; b < cfg.depth; ++b) {
        const std::string p = "v.blk." + std::to_string(b) + ".";
        auto& B = w.blocks[b];
        B.norm1_w = up(g, p + "ln1.weight", H, w, err); if (!B.norm1_w) return false;
        B.norm1_b = up(g, p + "ln1.bias",   H, w, err); if (!B.norm1_b) return false;
        B.qkv_w   = up(g, p + "attn_qkv.weight", 3 * H * H, w, err); if (!B.qkv_w) return false;
        B.qkv_b   = up(g, p + "attn_qkv.bias",   3 * H,     w, err); if (!B.qkv_b) return false;
        B.proj_w  = up(g, p + "attn_out.weight", H * H, w, err); if (!B.proj_w) return false;
        B.proj_b  = up(g, p + "attn_out.bias",   H,     w, err); if (!B.proj_b) return false;
        B.norm2_w = up(g, p + "ln2.weight", H, w, err); if (!B.norm2_w) return false;
        B.norm2_b = up(g, p + "ln2.bias",   H, w, err); if (!B.norm2_b) return false;
        B.fc1_w   = up(g, p + "ffn_up.weight",   I * H, w, err); if (!B.fc1_w) return false;
        B.fc1_b   = up(g, p + "ffn_up.bias",     I,     w, err); if (!B.fc1_b) return false;
        B.fc2_w   = up(g, p + "ffn_down.weight", H * I, w, err); if (!B.fc2_w) return false;
        B.fc2_b   = up(g, p + "ffn_down.bias",   H,     w, err); if (!B.fc2_b) return false;
    }

    // llama.cpp calls the merger's pre-norm `v.post_ln` and numbers the projection's two linears
    // by their position in the original nn.Sequential, so mm.0 is fc1 and mm.2 is fc2 (mm.1 was
    // the activation).
    w.merger_norm_w = up(g, "v.post_ln.weight", H, w, err); if (!w.merger_norm_w) return false;
    w.merger_norm_b = up(g, "v.post_ln.bias",   H, w, err); if (!w.merger_norm_b) return false;
    w.merger_fc1_w  = up(g, "mm.0.weight", M * M, w, err);  if (!w.merger_fc1_w) return false;
    w.merger_fc1_b  = up(g, "mm.0.bias",   M,     w, err);  if (!w.merger_fc1_b) return false;
    w.merger_fc2_w  = up(g, "mm.2.weight", (long)cfg.out_hidden * M, w, err);
    if (!w.merger_fc2_w) return false;
    w.merger_fc2_b  = up(g, "mm.2.bias", cfg.out_hidden, w, err);
    if (!w.merger_fc2_b) return false;
    return true;
}

}  // namespace sparkinfer
