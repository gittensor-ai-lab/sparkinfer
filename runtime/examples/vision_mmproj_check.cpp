// Does the mmproj GGUF loader produce the same vision tower as the safetensors checkpoint?
//
// Both loaders fill one QwenVisionWeights, so the check is direct: load the tower twice from two
// different files and compare every buffer. This is the same discipline the ternary work needed --
// validate against the ORIGINAL, not against your own encoder -- because a loader that reads a
// wrong name, a transposed matrix or an interleave backwards still produces a tower that runs.
//
//   vision_mmproj_check <mmproj.gguf> <safetensors_dir>
//
// An mmproj may be quantized (Q8_0 against the checkpoint's bf16), so exact equality is the wrong
// bar; cosine is the right one. A correct load of a Q8_0 mmproj lands at ~0.999+. Anything that
// reads the wrong tensor lands near zero, and anything transposed lands somewhere obviously wrong.
#include "sparkinfer/gguf.h"
#include "sparkinfer/safetensors.h"
#include "sparkinfer/models/qwen_vision.h"
#include "sparkinfer/models/qwen_vision_hf_config.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;
using bf16 = unsigned short;

static float bf16_to_f32(bf16 h) { unsigned u = (unsigned)h << 16; float f; std::memcpy(&f, &u, 4); return f; }

static int failures = 0;

// Cosine between two device bf16 buffers of n values, plus the largest absolute difference.
static void compare(const char* what, const void* a, const void* b, long n) {
    if (!a || !b) { std::printf("  %-34s MISSING (%p / %p)\n", what, a, b); failures++; return; }
    std::vector<bf16> ha((size_t)n), hb((size_t)n);
    cudaMemcpy(ha.data(), a, (size_t)n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, (size_t)n * 2, cudaMemcpyDeviceToHost);
    double dot = 0, na = 0, nb = 0, maxd = 0;
    for (long i = 0; i < n; ++i) {
        const double x = bf16_to_f32(ha[i]), y = bf16_to_f32(hb[i]);
        dot += x * y; na += x * x; nb += y * y;
        const double d = std::fabs(x - y);
        if (d > maxd) maxd = d;
    }
    const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    const bool ok = cos >= 0.99;
    std::printf("  %-34s cos %+.6f  maxdiff %.4g  %s\n", what, cos, maxd, ok ? "" : "<-- WRONG");
    if (!ok) failures++;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: vision_mmproj_check <mmproj.gguf> <safetensors_dir>\n"); return 2; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("vision_mmproj_check: SKIPPED (no CUDA device)\n"); return 0;
    }

    GGUF g;
    if (!g.open(argv[1])) { std::printf("cannot open %s\n", argv[1]); return 1; }
    QwenVisionConfig gcfg;
    if (!qwen_vision_config_from_gguf(g, gcfg)) { std::printf("not a vision mmproj\n"); return 1; }
    std::printf("mmproj config: depth=%d hidden=%d heads=%d ffn=%d patch=%d merge=%d out=%d pos=%d\n",
                gcfg.depth, gcfg.hidden, gcfg.n_heads, gcfg.intermediate, gcfg.patch_size,
                gcfg.spatial_merge, gcfg.out_hidden, gcfg.num_pos_embeddings);

    std::string err;
    QwenVisionWeights gw;
    if (!load_qwen_vision_weights_gguf(g, gcfg, gw, err)) {
        std::printf("mmproj load failed: %s\n", err.c_str()); return 1;
    }
    std::printf("mmproj loaded: %zu blocks, %zu device buffers\n", gw.blocks.size(), gw.owned.size());

    SafeTensorsModel st;
    if (!st.open(argv[2])) { std::printf("cannot open checkpoint %s\n", argv[2]); return 1; }
    QwenVisionConfig scfg;
    if (!qwen_vision_config_from_hf_json(argv[2], scfg)) {
        std::printf("no vision_config in %s -- using the mmproj's own geometry\n", argv[2]);
        scfg = gcfg;
    }
    QwenVisionWeights sw;
    if (!load_qwen_vision_weights(st, scfg, sw, err)) {
        std::printf("checkpoint load failed: %s\n", err.c_str()); return 1;
    }

    if (scfg.depth != gcfg.depth || scfg.hidden != gcfg.hidden) {
        std::printf("geometry disagrees: checkpoint depth=%d hidden=%d, mmproj depth=%d hidden=%d\n",
                    scfg.depth, scfg.hidden, gcfg.depth, gcfg.hidden);
        return 1;
    }

    const long H = gcfg.hidden, I = gcfg.intermediate, M = gcfg.merged_patch_dim();
    const long patch_in = (long)gcfg.in_channels * gcfg.temporal_patch * gcfg.patch_size * gcfg.patch_size;

    std::printf("--- patch embedding (the interleave is what this catches)\n");
    compare("patch_w", gw.patch_w, sw.patch_w, H * patch_in);
    compare("patch_b", gw.patch_b, sw.patch_b, H);

    std::printf("--- position table (host side)\n");
    {
        const size_t n = gw.pos_table.size();
        double dot = 0, na = 0, nb = 0;
        const bool same = n == sw.pos_table.size();
        for (size_t i = 0; same && i < n; ++i) {
            dot += (double)gw.pos_table[i] * sw.pos_table[i];
            na += (double)gw.pos_table[i] * gw.pos_table[i];
            nb += (double)sw.pos_table[i] * sw.pos_table[i];
        }
        if (!same) { std::printf("  pos_table SIZE MISMATCH %zu vs %zu\n", n, sw.pos_table.size()); failures++; }
        else {
            const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
            std::printf("  %-34s cos %+.6f  (%d x %d)  %s\n", "pos_table", cos, gw.pos_side,
                        gw.pos_side, cos >= 0.99 ? "" : "<-- WRONG");
            if (cos < 0.99) failures++;
        }
    }

    std::printf("--- blocks (first, middle, last)\n");
    for (int b : {0, gcfg.depth / 2, gcfg.depth - 1}) {
        const auto& G = gw.blocks[b];
        const auto& S = sw.blocks[b];
        char tag[64];
        std::snprintf(tag, sizeof tag, "blk.%d.attn_qkv.weight", b);  compare(tag, G.qkv_w, S.qkv_w, 3 * H * H);
        std::snprintf(tag, sizeof tag, "blk.%d.attn_qkv.bias", b);    compare(tag, G.qkv_b, S.qkv_b, 3 * H);
        std::snprintf(tag, sizeof tag, "blk.%d.attn_out.weight", b);  compare(tag, G.proj_w, S.proj_w, H * H);
        std::snprintf(tag, sizeof tag, "blk.%d.ffn_up.weight", b);    compare(tag, G.fc1_w, S.fc1_w, I * H);
        std::snprintf(tag, sizeof tag, "blk.%d.ffn_down.weight", b);  compare(tag, G.fc2_w, S.fc2_w, H * I);
        std::snprintf(tag, sizeof tag, "blk.%d.ln1.weight", b);       compare(tag, G.norm1_w, S.norm1_w, H);
        std::snprintf(tag, sizeof tag, "blk.%d.ln2.bias", b);         compare(tag, G.norm2_b, S.norm2_b, H);
    }

    std::printf("--- merger\n");
    compare("merger_norm_w", gw.merger_norm_w, sw.merger_norm_w, H);
    compare("merger_fc1_w",  gw.merger_fc1_w,  sw.merger_fc1_w,  M * M);
    compare("merger_fc1_b",  gw.merger_fc1_b,  sw.merger_fc1_b,  M);
    compare("merger_fc2_w",  gw.merger_fc2_w,  sw.merger_fc2_w,  (long)gcfg.out_hidden * M);
    compare("merger_fc2_b",  gw.merger_fc2_b,  sw.merger_fc2_b,  gcfg.out_hidden);

    free_qwen_vision_weights(gw);
    free_qwen_vision_weights(sw);
    std::printf("vision_mmproj_check: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
