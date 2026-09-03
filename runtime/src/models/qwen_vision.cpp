// Vision tower forward: patch embed -> position embed -> N blocks -> 2x2 merger.
//
// Validated against bench/scripts/vision_ref.py, which was itself validated against the released
// weights and the transformers reference (see the stage commits on this branch). The reference is
// the contract: if this file and vision_ref.py disagree, this file is wrong.
#include "sparkinfer/models/qwen_vision.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "sparkinfer/safetensors.h"
#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/vision.h"

namespace sparkinfer {

namespace {

using bf16 = unsigned short;

bool cu_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

float bf16_to_f32(bf16 h) { unsigned u = (unsigned)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
bf16 f32_to_bf16(float f) { unsigned u; std::memcpy(&u, &f, 4); return (bf16)(u >> 16); }

// Upload a checkpoint tensor verbatim. Everything under model.visual.* is already bf16, so this
// is a straight copy -- no dequant, no requant, no layout change.
const void* upload(SafeTensorsModel& st, const std::string& name, long want_values,
                   QwenVisionWeights& w, std::string& err) {
    const STTensor* t = st.tensor(name);
    if (!t) { err = "missing vision tensor " + name; return nullptr; }
    if (t->dtype != STDType::BF16) { err = name + ": expected BF16"; return nullptr; }
    if (want_values > 0 && t->n_values != want_values) {
        err = name + ": expected " + std::to_string(want_values) + " values, got "
            + std::to_string(t->n_values);
        return nullptr;
    }
    void* d = nullptr;
    if (!cu_ok(cudaMalloc(&d, (size_t)t->n_values * sizeof(bf16)), ("cudaMalloc " + name).c_str(), err))
        return nullptr;
    if (!cu_ok(cudaMemcpy(d, t->data, (size_t)t->n_values * sizeof(bf16), cudaMemcpyHostToDevice),
               ("upload " + name).c_str(), err)) { cudaFree(d); return nullptr; }
    w.owned.push_back(d);
    return d;
}

}  // namespace

bool load_qwen_vision_weights(SafeTensorsModel& st, const QwenVisionConfig& cfg,
                              QwenVisionWeights& w, std::string& err) {
    const long H = cfg.hidden, I = cfg.intermediate;
    const long patch_in = (long)cfg.in_channels * cfg.temporal_patch * cfg.patch_size * cfg.patch_size;
    const long M = cfg.merged_patch_dim();

    w.patch_w = upload(st, "model.visual.patch_embed.proj.weight", H * patch_in, w, err);
    if (!w.patch_w) return false;
    w.patch_b = upload(st, "model.visual.patch_embed.proj.bias", H, w, err);
    if (!w.patch_b) return false;

    // Position table stays on the host: the bilinear resample depends on the image's patch grid,
    // so it is recomputed and uploaded per image rather than baked in here.
    const STTensor* pos = st.tensor("model.visual.pos_embed.weight");
    if (!pos || pos->n_values != (long)cfg.num_pos_embeddings * H) {
        err = "model.visual.pos_embed.weight missing or wrong size"; return false;
    }
    w.pos_side = (int)llround(std::sqrt((double)cfg.num_pos_embeddings));
    if ((long)w.pos_side * w.pos_side != cfg.num_pos_embeddings) {
        err = "num_position_embeddings is not a perfect square"; return false;
    }
    w.pos_table.resize((size_t)pos->n_values);
    for (long i = 0; i < pos->n_values; i++)
        w.pos_table[i] = bf16_to_f32(((const bf16*)pos->data)[i]);

    w.blocks.resize(cfg.depth);
    for (int b = 0; b < cfg.depth; b++) {
        const std::string p = "model.visual.blocks." + std::to_string(b) + ".";
        auto& B = w.blocks[b];
        B.norm1_w = upload(st, p + "norm1.weight", H, w, err); if (!B.norm1_w) return false;
        B.norm1_b = upload(st, p + "norm1.bias",   H, w, err); if (!B.norm1_b) return false;
        B.qkv_w   = upload(st, p + "attn.qkv.weight", 3 * H * H, w, err); if (!B.qkv_w) return false;
        B.qkv_b   = upload(st, p + "attn.qkv.bias",   3 * H,     w, err); if (!B.qkv_b) return false;
        B.proj_w  = upload(st, p + "attn.proj.weight", H * H, w, err); if (!B.proj_w) return false;
        B.proj_b  = upload(st, p + "attn.proj.bias",   H,     w, err); if (!B.proj_b) return false;
        B.norm2_w = upload(st, p + "norm2.weight", H, w, err); if (!B.norm2_w) return false;
        B.norm2_b = upload(st, p + "norm2.bias",   H, w, err); if (!B.norm2_b) return false;
        B.fc1_w   = upload(st, p + "mlp.linear_fc1.weight", I * H, w, err); if (!B.fc1_w) return false;
        B.fc1_b   = upload(st, p + "mlp.linear_fc1.bias",   I,     w, err); if (!B.fc1_b) return false;
        B.fc2_w   = upload(st, p + "mlp.linear_fc2.weight", H * I, w, err); if (!B.fc2_w) return false;
        B.fc2_b   = upload(st, p + "mlp.linear_fc2.bias",   H,     w, err); if (!B.fc2_b) return false;
    }

    w.merger_norm_w = upload(st, "model.visual.merger.norm.weight", H, w, err); if (!w.merger_norm_w) return false;
    w.merger_norm_b = upload(st, "model.visual.merger.norm.bias",   H, w, err); if (!w.merger_norm_b) return false;
    w.merger_fc1_w  = upload(st, "model.visual.merger.linear_fc1.weight", M * M, w, err); if (!w.merger_fc1_w) return false;
    w.merger_fc1_b  = upload(st, "model.visual.merger.linear_fc1.bias",   M,     w, err); if (!w.merger_fc1_b) return false;
    w.merger_fc2_w  = upload(st, "model.visual.merger.linear_fc2.weight", (long)cfg.out_hidden * M, w, err);
    if (!w.merger_fc2_w) return false;
    w.merger_fc2_b  = upload(st, "model.visual.merger.linear_fc2.bias", cfg.out_hidden, w, err);
    if (!w.merger_fc2_b) return false;
    return true;
}

void free_qwen_vision_weights(QwenVisionWeights& w) {
    for (void* p : w.owned) cudaFree(p);
    w.owned.clear();
    w.blocks.clear();
    w.pos_table.clear();
}

bool qwen_vision_forward(const QwenVisionWeights& w, const QwenVisionConfig& cfg,
                         const float* pixels_host, int grid_h, int grid_w,
                         float* out_host, std::string& err) {
    const int H = cfg.hidden, I = cfg.intermediate, heads = cfg.n_heads, hd = cfg.head_dim();
    const int merge = cfg.spatial_merge;
    const int N = grid_h * grid_w;
    const int patch_in = cfg.in_channels * cfg.temporal_patch * cfg.patch_size * cfg.patch_size;
    const int M = cfg.merged_patch_dim(), nblk = (grid_h / merge) * (grid_w / merge);
    if (grid_h % merge || grid_w % merge) { err = "patch grid must divide by spatial_merge"; return false; }

    cudaStream_t s = nullptr;
    std::vector<void*> tmp;
    auto alloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        tmp.push_back(p); return p;
    };
    auto cleanup = [&]() { for (void* p : tmp) cudaFree(p); tmp.clear(); };

    void* d_pix  = alloc((size_t)N * patch_in * sizeof(bf16));
    void* d_x    = alloc((size_t)N * H * sizeof(bf16));
    void* d_h    = alloc((size_t)N * H * sizeof(bf16));
    void* d_qkv  = alloc((size_t)N * 3 * H * sizeof(bf16));
    void* d_q    = alloc((size_t)N * H * sizeof(bf16));
    void* d_k    = alloc((size_t)N * H * sizeof(bf16));
    void* d_v    = alloc((size_t)N * H * sizeof(bf16));
    void* d_att  = alloc((size_t)N * H * sizeof(bf16));
    void* d_mlp  = alloc((size_t)N * I * sizeof(bf16));
    void* d_mrg  = alloc((size_t)nblk * M * sizeof(bf16));
    void* d_out  = alloc((size_t)nblk * (size_t)cfg.out_hidden * sizeof(bf16));
    if (!d_pix || !d_x || !d_h || !d_qkv || !d_q || !d_k || !d_v || !d_att || !d_mlp || !d_mrg || !d_out) {
        cleanup(); err = "vision forward: device allocation failed"; return false;
    }

    // --- patch embed: [N, patch_in] @ [H, patch_in]^T -> [N, H] ---
    {
        std::vector<bf16> hp((size_t)N * patch_in);
        for (size_t i = 0; i < hp.size(); i++) hp[i] = f32_to_bf16(pixels_host[i]);
        if (!cu_ok(cudaMemcpy(d_pix, hp.data(), hp.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                   "upload pixels", err)) { cleanup(); return false; }
    }
    kernels::launch_prefill_gemm(d_pix, w.patch_w, d_x, N, H, patch_in, s);
    kernels::launch_vision_add_bias(d_x, w.patch_b, N, H, s);

    // --- position embed: bilinear resample of the pos_side x pos_side table onto this grid ---
    // Host-side and uploaded, matching vision_ref.py exactly. At grid == pos_side this is the
    // identity, which is the shape the forward is first validated at.
    {
        const int side = w.pos_side;
        std::vector<bf16> hp((size_t)N * H);
        for (int gy = 0; gy < grid_h; gy++) {
            const float fy = grid_h > 1 ? (float)gy * (side - 1) / (grid_h - 1) : 0.f;
            const int y0 = (int)floorf(fy), y1 = y0 + 1 < side ? y0 + 1 : side - 1;
            const float wy = fy - y0;
            for (int gx = 0; gx < grid_w; gx++) {
                const float fx = grid_w > 1 ? (float)gx * (side - 1) / (grid_w - 1) : 0.f;
                const int x0 = (int)floorf(fx), x1 = x0 + 1 < side ? x0 + 1 : side - 1;
                const float wx = fx - x0;
                const float* p00 = &w.pos_table[((size_t)y0 * side + x0) * H];
                const float* p01 = &w.pos_table[((size_t)y0 * side + x1) * H];
                const float* p10 = &w.pos_table[((size_t)y1 * side + x0) * H];
                const float* p11 = &w.pos_table[((size_t)y1 * side + x1) * H];
                bf16* dst = &hp[((size_t)gy * grid_w + gx) * H];
                for (int d = 0; d < H; d++)
                    dst[d] = f32_to_bf16(p00[d] * (1 - wy) * (1 - wx) + p01[d] * (1 - wy) * wx
                                       + p10[d] * wy * (1 - wx) + p11[d] * wy * wx);
            }
        }
        if (!cu_ok(cudaMemcpy(d_h, hp.data(), hp.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                   "upload pos embed", err)) { cleanup(); return false; }
    }
    kernels::launch_vision_residual_add(d_x, d_h, (long)N * H, s);

    const float scale = 1.0f / std::sqrt((float)hd);
    for (int b = 0; b < cfg.depth; b++) {
        const auto& B = w.blocks[b];
        kernels::launch_vision_layernorm(d_x, B.norm1_w, B.norm1_b, d_h, N, H, 1e-6f, s);
        kernels::launch_prefill_gemm(d_h, B.qkv_w, d_qkv, N, 3 * H, H, s);
        kernels::launch_vision_add_bias(d_qkv, B.qkv_b, N, 3 * H, s);
        // Split the fused [N, 3H] into q/k/v. cudaMemcpy2D does the strided extraction, so the
        // attention kernel can assume a contiguous [N, heads*hd] layout.
        const size_t row = (size_t)3 * H * sizeof(bf16), col = (size_t)H * sizeof(bf16);
        cudaMemcpy2DAsync(d_q, col, (const char*)d_qkv,             row, col, N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(d_k, col, (const char*)d_qkv + col,       row, col, N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(d_v, col, (const char*)d_qkv + 2 * col,   row, col, N, cudaMemcpyDeviceToDevice, s);
        kernels::launch_vision_attention(d_q, d_k, d_v, d_att, N, heads, hd, scale, s);
        kernels::launch_prefill_gemm(d_att, B.proj_w, d_h, N, H, H, s);
        kernels::launch_vision_add_bias(d_h, B.proj_b, N, H, s);
        kernels::launch_vision_residual_add(d_x, d_h, (long)N * H, s);

        kernels::launch_vision_layernorm(d_x, B.norm2_w, B.norm2_b, d_h, N, H, 1e-6f, s);
        kernels::launch_prefill_gemm(d_h, B.fc1_w, d_mlp, N, I, H, s);
        kernels::launch_vision_add_bias(d_mlp, B.fc1_b, N, I, s);
        kernels::launch_vision_gelu_tanh(d_mlp, (long)N * I, s);
        kernels::launch_prefill_gemm(d_mlp, B.fc2_w, d_h, N, H, I, s);
        kernels::launch_vision_add_bias(d_h, B.fc2_b, N, H, s);
        kernels::launch_vision_residual_add(d_x, d_h, (long)N * H, s);
    }

    // --- merger ---
    kernels::launch_vision_layernorm(d_x, w.merger_norm_w, w.merger_norm_b, d_h, N, H, 1e-6f, s);
    kernels::launch_vision_patch_merge(d_h, d_mrg, grid_h, grid_w, merge, H, s);
    kernels::launch_prefill_gemm(d_mrg, w.merger_fc1_w, d_out, nblk, M, M, s);
    kernels::launch_vision_add_bias(d_out, w.merger_fc1_b, nblk, M, s);
    kernels::launch_vision_gelu_tanh(d_out, (long)nblk * M, s);
    // fc1 wrote M-wide into d_out; fc2 reads that and writes out_hidden-wide. Bounce through
    // d_mrg so the GEMM's input and output never alias.
    if (!cu_ok(cudaMemcpyAsync(d_mrg, d_out, (size_t)nblk * M * sizeof(bf16),
                               cudaMemcpyDeviceToDevice, s), "merger bounce", err)) { cleanup(); return false; }
    kernels::launch_prefill_gemm(d_mrg, w.merger_fc2_w, d_out, nblk, cfg.out_hidden, M, s);
    kernels::launch_vision_add_bias(d_out, w.merger_fc2_b, nblk, cfg.out_hidden, s);

    if (!cu_ok(cudaStreamSynchronize(s), "vision forward sync", err)) { cleanup(); return false; }
    {
        std::vector<bf16> ho((size_t)nblk * cfg.out_hidden);
        if (!cu_ok(cudaMemcpy(ho.data(), d_out, ho.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
                   "download vision out", err)) { cleanup(); return false; }
        for (size_t i = 0; i < ho.size(); i++) out_host[i] = bf16_to_f32(ho[i]);
    }
    cleanup();
    return true;
}

}  // namespace sparkinfer
