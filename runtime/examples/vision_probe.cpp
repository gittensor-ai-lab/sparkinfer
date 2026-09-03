// Does this checkpoint carry a usable vision tower, and does it match its own config?
//
// Stage 1 of image support. Deliberately does NO inference: it reads config.json's vision_config,
// then walks the safetensors shards and checks that every tensor the tower needs is present with
// the shape the config implies. Getting this wrong silently -- a missing block, a transposed
// weight, a patch embed the reader skipped -- is how vision integrations produce plausible
// garbage, so the shapes are asserted against the config rather than merely reported.
//
// Usage: vision_probe <checkpoint_dir>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sparkinfer/safetensors.h"
#include "sparkinfer/models/qwen_vision_config.h"
#include "sparkinfer/models/qwen_vision_hf_config.h"

using sparkinfer::QwenVisionConfig;
using sparkinfer::SafeTensorsModel;
using sparkinfer::STTensor;

namespace {

int g_fail = 0;

const STTensor* find(SafeTensorsModel& st, const std::string& name) { return st.tensor(name); }

// Checks presence AND shape. `want` is the config-derived expectation; -1 means "don't care".
void expect(SafeTensorsModel& st, const std::string& name, std::vector<long> want) {
    const STTensor* t = find(st, name);
    if (!t) { printf("  MISSING  %s\n", name.c_str()); g_fail++; return; }
    bool ok = (int)want.size() == t->n_dims;
    for (size_t i = 0; ok && i < want.size(); i++)
        if (want[i] >= 0 && want[i] != t->dims[i]) ok = false;
    if (!ok) {
        printf("  SHAPE    %s got [", name.c_str());
        for (int i = 0; i < t->n_dims; i++) printf("%ld%s", t->dims[i], i + 1 < t->n_dims ? ", " : "");
        printf("] want [");
        for (size_t i = 0; i < want.size(); i++)
            printf("%s%s", want[i] < 0 ? "*" : std::to_string(want[i]).c_str(), i + 1 < want.size() ? ", " : "");
        printf("]\n");
        g_fail++;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <checkpoint_dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];

    QwenVisionConfig vc;
    std::string err;
    if (!qwen_vision_config_from_hf_json(dir, vc, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    if (!vc.present) { printf("[FAIL] no vision_config -- text-only checkpoint\n"); return 1; }

    printf("vision_config: depth=%d hidden=%d heads=%d (head_dim=%d) inter=%d\n",
           vc.depth, vc.hidden, vc.n_heads, vc.head_dim(), vc.intermediate);
    printf("               patch=%d temporal=%d merge=%d -> merged_dim=%d out_hidden=%d\n",
           vc.patch_size, vc.temporal_patch, vc.spatial_merge, vc.merged_patch_dim(), vc.out_hidden);
    printf("               act=%s pos_embeddings=%d granularity=%d px\n",
           vc.hidden_act.c_str(), vc.num_pos_embeddings, vc.size_granularity());
    printf("               tokens: vision_start=%d image=%d vision_end=%d\n",
           vc.vision_start_token_id, vc.image_token_id, vc.vision_end_token_id);
    printf("               norm: mean=%.2f std=%.2f  pixels=[%ld, %ld]\n",
           vc.mean[0], vc.std_[0], vc.min_pixels, vc.max_pixels);

    SafeTensorsModel st;
    if (!st.open(dir)) { printf("[FAIL] could not open safetensors model in %s\n", dir.c_str()); return 1; }

    const int H = vc.hidden, I = vc.intermediate;
    // Conv3d patch embed: [out, in_ch, t_patch, ph, pw]. This is the 5-D tensor the reader used to
    // skip outright -- if it is missing here, the dims[5] widening did not take.
    expect(st, "model.visual.patch_embed.proj.weight",
           {H, vc.in_channels, vc.temporal_patch, vc.patch_size, vc.patch_size});
    expect(st, "model.visual.patch_embed.proj.bias", {H});
    expect(st, "model.visual.pos_embed.weight", {vc.num_pos_embeddings, H});

    for (int b = 0; b < vc.depth; b++) {
        const std::string p = "model.visual.blocks." + std::to_string(b) + ".";
        expect(st, p + "norm1.weight", {H});
        expect(st, p + "norm1.bias",   {H});
        expect(st, p + "attn.qkv.weight", {3 * H, H});
        expect(st, p + "attn.qkv.bias",   {3 * H});
        expect(st, p + "attn.proj.weight", {H, H});
        expect(st, p + "attn.proj.bias",   {H});
        expect(st, p + "norm2.weight", {H});
        expect(st, p + "norm2.bias",   {H});
        expect(st, p + "mlp.linear_fc1.weight", {I, H});
        expect(st, p + "mlp.linear_fc1.bias",   {I});
        expect(st, p + "mlp.linear_fc2.weight", {H, I});
        expect(st, p + "mlp.linear_fc2.bias",   {H});
    }

    // Merger: norm over one patch, then 2x2-merged patches (4608) -> LM hidden (5120).
    expect(st, "model.visual.merger.norm.weight", {H});
    expect(st, "model.visual.merger.norm.bias",   {H});
    expect(st, "model.visual.merger.linear_fc1.weight", {vc.merged_patch_dim(), vc.merged_patch_dim()});
    expect(st, "model.visual.merger.linear_fc1.bias",   {vc.merged_patch_dim()});
    expect(st, "model.visual.merger.linear_fc2.weight", {vc.out_hidden, vc.merged_patch_dim()});
    expect(st, "model.visual.merger.linear_fc2.bias",   {vc.out_hidden});

    // The merger's output must equal the LM hidden size or the embeddings cannot be spliced in.
    // Checked here rather than assumed: a mismatch is the whole feature failing, quietly.
    const STTensor* emb = find(st, "model.language_model.embed_tokens.weight");
    if (!emb) emb = find(st, "model.embed_tokens.weight");
    if (emb && emb->n_dims == 2 && emb->dims[1] != vc.out_hidden) {
        printf("  MISMATCH merger out_hidden=%d but LM embed dim=%ld\n", vc.out_hidden, emb->dims[1]);
        g_fail++;
    } else if (emb) {
        printf("LM embed dim %ld matches merger out_hidden %d\n", emb->dims[1], vc.out_hidden);
    }

    const int expected = 3 + vc.depth * 12 + 6;
    printf("\nchecked %d vision tensors across %d blocks\n", expected, vc.depth);
    if (g_fail) { printf("[FAIL] %d problem(s)\n", g_fail); return 1; }
    printf("[OK] vision tower is complete and matches its config\n");
    return 0;
}
